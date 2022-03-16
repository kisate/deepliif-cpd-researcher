import os
root_folder = os.getenv("DATA_DIR")

import subprocess
print('Install packages and start gpu monitor ...')
subprocess.run('pip install dominate visdom gpustat numba==0.54.1 --user',shell=True)
subprocess.run('pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html --user',shell=True)
subprocess.Popen('bash monitor_gpu.sh',shell=True)

os.environ['VOLUME_DISPLAY_NAME'] = 'AdditionalDeepLIIFVolume'

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