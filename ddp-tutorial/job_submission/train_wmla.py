from __future__ import division

import subprocess

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision
from torchvision import transforms

import numpy as np

import os

def init_process():
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://' + os.environ['MASTER_ADDR'] + ':' + os.environ['MASTER_PORT'],
        rank=int(os.environ['RANK']),
        world_size=int(os.environ['WORLD_SIZE']))
            
def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    

def trainEpoch(train_loader, model, criterion, optimizer, epoch, device):

    # object to store & plot the losses
    losses = AverageMeter()

    # switch to train mode
    model.train()

    # Train in mini-batches
    for batch_idx, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.update(loss.data.cpu().numpy(), labels.size(0))
        loss.backward()
        optimizer.step()

        # Print info
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
        #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
        #     epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), loss=losses))


def valEpoch(val_loader, model, criterion, epoch, device):

    losses = AverageMeter()

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():

        # Mini-batches
        for batch_idx, data in enumerate(val_loader):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.update(loss.data.cpu().numpy(), labels.size(0))
            _, predicted = torch.max(outputs, 1)


            # Save predicted to compute accuracy
            if batch_idx==0:
                out = predicted.data.cpu().numpy()
                label = labels.cpu().numpy()
            else:
                out = np.concatenate((out,predicted.data.cpu().numpy()),axis=0)
                label = np.concatenate((label, labels.cpu().numpy()),axis=0)

        # Accuracy
        acc = np.sum(out == label)/len(out)

        # Print validation info
        print('Validation set: Average loss: {:.4f}\t'
              'Accuracy {acc}'.format(losses.avg, acc=acc))

        # Return acc as the validation outcome
        return acc

def prepare_data(rank, world_size):
    data_path = os.environ["DRIVE_DATA_PATH"]

    # CIFAR-10 Data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           download=False, transform=transform)
 
    trainloader = prepare(trainset, rank, world_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                             shuffle=False, num_workers=0)
    
    return trainloader, testloader


def save_model(model):
    model_path = os.environ["DRIVE_MODEL_PATH"]
    
    os.makedirs(model_path,exist_ok=True)
    if os.getenv('APP_ID') is not None:
        os.makedirs(f'{model_path}/{os.environ["APP_ID"]}',exist_ok=True)
        path = f'{model_path}/{os.environ["APP_ID"]}/model.pt'
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.module.state_dict(), f'{model_path}/model.pt')
    
    
def trainProcess():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ['WORLD_SIZE'])
    
    print(f"RANK: {rank} {local_rank}")
    
    # instantiate the model and move it to the right device
    model = Net().to(local_rank)
    
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    trainloader, testloader = prepare_data(rank, world_size)
    
    best_val = float(0)

    
    # Now, let's start the training process!
    print('Training...')
    for epoch in range(10):
        trainloader.sampler.set_epoch(epoch)  

        # Compute a training epoch
        trainEpoch(trainloader, model, criterion, optimizer, epoch, local_rank)

        if rank == 0:
            # Compute a validation epoch
            lossval = valEpoch(testloader, model.module, criterion, epoch, local_rank)

            # Print validation accuracy and best validation accuracy
            best_val = max(lossval, best_val)
            print('** Validation: %f (best) - %f (current)' % (best_val, lossval))
            
        dist.barrier()
    
    if rank == 0:
        save_model(model)



if __name__ == "__main__":
    # Init DDP
    print('------ initiate process group... ------')
    init_process()
    
    # Training process
    trainProcess()
    
    dist.destroy_process_group()