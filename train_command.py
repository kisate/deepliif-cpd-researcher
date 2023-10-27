import os
from pathlib import Path
DOWNLOAD_DATA = False

root_folder = os.getenv("DATA_DIR")
Path(root_folder).mkdir(parents=True, exist_ok=True)

print(os.listdir(root_folder))


import subprocess

if DOWNLOAD_DATA:
    print(subprocess.run(f"bash download_data.sh {str(Path(root_folder).parent)}", shell=True))

print(os.listdir('/gpfs/mydatafs/'))
print(os.listdir('/gpfs/'))
print(os.listdir('/gpfs/mygpfs'))
print(os.listdir('.'))
print(root_folder)
print(os.listdir(root_folder))
print(os.getcwd())
print(os.listdir('/gpfs/mydatafs/MNIST'))
print(os.listdir('/'))
print(os.listdir('/mnts'))
# print("HEH")

raise

print('Install packages and start gpu monitor ...')
subprocess.run('pip install dominate visdom gpustat numba --user',shell=True)
subprocess.run('pip install torch -f https://download.pytorch.org/whl/torch_stable.html --user',shell=True)
subprocess.Popen('bash monitor_gpu.sh',shell=True)

# os.environ['VOLUME_DISPLAY_NAME'] = 'cpd::demo-project-pvc'
os.environ['TORCH_HOME'] = os.environ['PWD'] # tell pytorch hub to use the working directory to cache pre-trained models, not $HOME which wmla users have no write access to
print(os.environ['TORCH_HOME'])

import psutil

if __name__ == '__main__':
    print('-------- start training... --------')
    subprocess.run(f'python cli.py train --dataroot {root_folder} --name Test_Model_wendy_wmla_$APP_ID --remote True --remote-transfer-cmd storage_volume_utils.upload --batch-size 3 --gpu-ids 0 --display-env $APP_ID',shell=True)

    # (DP) use batch_size > 1 if you want to leverage multiple gpus; batch_size=1 will only effectively use 1 gpu, because this setting is picked up by DP's single process and the amount is distributed across multiple gpus
    # (DDP) even if batch size is 1, each process/gpu will get one image per mini batch. because this setting is picked up by each of DDP's processes, one process having 1 gpu
    
    for process in psutil.process_iter():
        if process.cmdline() == ['bash', 'monitor_gpu.sh']:
            print('Terminating gpu monitor...')
            process.terminate()
            break