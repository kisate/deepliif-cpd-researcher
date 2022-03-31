from visdom import Visdom

import numpy as np
import pickle
import os
import storage_volume_utils as sv

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

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main',remote=True,wmla=True):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
        self.remote = remote
        self.pickle_dir = './checkpoints/pickle'
        self.wmla = wmla
        self.use_multi_proc = False
        self.rank = None
        
        if os.getenv('RANK') is not None:
            self.use_multi_proc = True
            self.rank = int(os.getenv('RANK'))
        elif os.getenv('LOCAL_RANK') is not None:
            self.use_multi_proc = True
            self.rank = int(os.getenv('LOCAL_RANK'))
            
        if os.getenv('APP_ID') is not None:
            self.pickle_dir = f'./checkpoints/{os.environ["APP_ID"]}/pickle'

        if self.remote:
             if (self.use_multi_proc and self.rank == 0) or (not self.use_multi_proc):
                os.makedirs(self.pickle_dir,exist_ok=True)
                path = os.path.join(self.pickle_dir,'env_name.pickle')
                pickle.dump(env_name,open(path,'wb'))
                print('Remote mode, snapshot created: env_name')
                if self.wmla:
                    sv.upload(path)

    def plot(self, var_name, split_name, title_name, x, y):
        if self.remote:
            path = os.path.join(self.pickle_dir,'plot.pickle')
            pickle.dump({'var_name':var_name,
                         'split_name':split_name,
                         'title_name':title_name,
                         'x':x,
                         'y':y},open(path,'wb'))
            print('Remote mode, snapshot refreshed: plot')
            if self.wmla:
                if (self.use_multi_proc and self.rank == 0) or (not self.use_multi_proc):
                    sv.upload(path)
        else:
            if var_name not in self.plots:
                self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel='Epochs',
                    ylabel=var_name
                ))
            else:
                self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')