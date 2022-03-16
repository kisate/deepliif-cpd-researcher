import os
root_folder = os.getenv("DATA_DIR")

import subprocess
print('Install packages and start gpu monitor ...')
subprocess.run('pip install dominate visdom gpustat numba==0.54.1 --user',shell=True)
subprocess.run('pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html --user',shell=True)
subprocess.run('pip install build --user',shell=True)
subprocess.run('python -m build',shell=True)
subprocess.run('pip install dist/deepliif-0.0.1.tar.gz --no-dependencies --user',shell=True)
os.environ['PATH'] = f"{os.environ['PATH']}:{os.environ['PYTHONUSERBASE']}/bin"
subprocess.Popen('bash monitor_gpu.sh',shell=True)

import psutil

if __name__ == '__main__':
    print('-------- start training... --------')
    subprocess.run(f'deepliif train --dataroot {root_folder} --name Test_Model_wendy_wmla --remote True --remote-transfer-cmd storage_volume_utils.upload --batch-size 3 --gpu-ids 0 --display-env $APP_ID',shell=True)
    print('-------- finished training --------')

    # (DP) use batch_size > 1 if you want to leverage multiple gpus; batch_size=1 will only effectively use 1 gpu, because this setting is picked up by DP's single process and the amount is distributed across multiple gpus
    # (DDP) even if batch size is 1, each process/gpu will get one image per mini batch. because this setting is picked up by each of DDP's processes, one process having 1 gpu
    
    for process in psutil.process_iter():
        if process.cmdline() == ['bash', 'monitor_gpu.sh']:
            print('Terminating gpu monitor...')
            process.terminate()
            break