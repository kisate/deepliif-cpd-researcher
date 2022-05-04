#!/usr/bin/env python

import redhareapiversion
from redhareapi import Kernel
import json
import os
import subprocess
import sys
from io import BytesIO
import base64
from PIL import Image

deploy_name = os.environ['REDHARE_MODEL_NAME']
dir_user = os.environ['REDHARE_MODEL_PATH']
path_model = 'model.pt'

dir_python_pkg = f"{dir_user}/python_packages"
os.makedirs(dir_python_pkg,exist_ok=True)
sys.path.insert(0, dir_python_pkg)

os.chdir(dir_user)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

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

class MatchKernel(Kernel):
    def __init__(self):
        self.model = None

    def on_kernel_start(self, kernel_context):
        
        #Kernel.log_debug("installing packages...")
        #out = subprocess.check_output(f'pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html --target={dir_python_pkg}', shell=True, text=True, stderr=subprocess.STDOUT)
        #Kernel.log_debug("finished installing packages")
        
        # load model
        model = Net()
        model.load_state_dict(torch.load(path_model))
        model.eval()
        self.model = model
        
    def on_task_invoke(self, task_context):
        try:
            Kernel.log_debug("on_task_invoke")

            while task_context != None:
                
                # parse input
                input_data = json.loads(task_context.get_input_data())
                img = Image.open(BytesIO(base64.b64decode(input_data['img'])))
                transform = transforms.ToTensor()
                input_tensor = transform(img)
                input_tensor = torch.stack([input_tensor])

                pred = self.model(input_tensor)
                _, pred_class = torch.max(pred, 1)
                pred_class = int(pred_class[0].numpy())
                
                output_data = {'pred_class':pred_class}
                task_context.set_output_data(json.dumps(output_data))
                task_context = task_context.next()
                    
        except Exception as e:
            Kernel.log_error(f"Failed due to {str(e)}")
    
    def on_kernel_shutdown(self):
        subprocess.run(f'rm -rf {dir_user}/*', shell=True)

        
if __name__ == '__main__':
    obj_kernel = MatchKernel()
    obj_kernel.run()
