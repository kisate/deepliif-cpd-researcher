from subprocess import Popen, PIPE
import time
import os
import pickle

cmd = 'python -m visdom.server'
Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
print('visdom started')

volume = 'demo_project_pvc'
app_id = 'cpd-40'

import utils


if __name__ == "__main__":

    # Plots
    global plotter
    
    path_init = f'/mnts/{volume}/checkpoints/{app_id}/pickle/env_name.pickle'
    while not os.path.exists(path_init):
        time.sleep(1)
    
    env_name = pickle.load(open(path_init,'rb'))
    plotter = utils.VisdomLinePlotter(env_name,remote=False)
    
    path_plot = f'/mnts/{volume}/checkpoints/{app_id}/pickle/plot.pickle'
    while not os.path.exists(path_plot):
        time.sleep(1)
    
    last_modified_time_plot = 0
    while True:
        last_modified_time_plot_new = os.path.getmtime(path_plot)
        if last_modified_time_plot_new > last_modified_time_plot:
            params_plot = pickle.load(open(path_plot,'rb'))
            plotter.plot(**params_plot)
            last_modified_time_plot = last_modified_time_plot_new
            print(f'plot refreshed, last modified time {last_modified_time_plot}')
        else:
            print('plot not refreshed')
        time.sleep(3)