# Researcher's Watson Studio Repository
This repository contains tutorials as well as Watson Studio specific files for DeepLIIF. It has Git submodules that point to clean release repositories. Refer to [Template Repository](https://github.com/drewmibm/watson-studio-template#watson-studio-template) for more information on how this set-up works, or instructions about how to set up Watson Studio projects.

Useful commands can be found in [cheat_sheet.txt](cheat_sheet.txt).

## Submodules
2 submodules are included:

- the original DeepLIIF repo (`deepliif-nadeemlab`)
- a forked repo (`deepliif-repo`)

You can use either one, depending on the access you have; or, add your own fork as a submodule.

## Tutorial Folders
Each tutorial folder contains examples about how to do a specific task. You don't really need them if you are already familiar with those, or simply want to train DeepLIIF model with modified code.

## Train DeepLIIF
You need:

- a folder with DeepLIIF code (we use `deepliif-repo`)
- a script for training command (we use `train_command.py` or `train_command_consumer_local.py`, where the later builds the deepliif package and runs deepliif commands for training; you can create your own based on it)
- cli commands to submit training (we use `WMLA_Job_Submission_CLI.ipynb` or `A1_Submit_WMLA_Training.ipynb`)

Once you've done modifications to the code in the deepliif submodule folder, occasionally you will change the `train_command.py` script (e.g., add additional package, change subprocess command, etc.), afterwards run commands similar to those in either of the two notebooks to submit the training job to WMLA.

Make sure you have utility script `storage_volume_utils.py` in the same folder as `train_command.py`, and in `train_command.py` you added argument `--remote-transfer-cmd storage_volume_utils.upload` to DeepLIIF's training command (`python cli.py train`, `deepliif train`, `python cli.py trainlaunch`, `deepliif trainlaunch`). This upload method helps transfer files in the WMLA training environment to a Cloud Pak for Data storage volume you have access to.

Output:

- **WMLA training log**: accessible using WMLA cli (as showcased in `WMLA_Job_Submission_CLI.ipynb`), or from WMLA web console
- **DeepLIIF's trained model/checkpoints**: it will be saved to a storage volume that you have access to; the location where the trained model is saved is controlled by 2 parameters:
  - environment variable `VOLUME_DISPLAY_NAME` in `train_command.py`: this value tells the data transfer code (specified by `--remote-transfer-cmd storage_volume_utils.upload` in DeepLIIF's training command) which storage volume to save the data to
  - `--checkpoints-dir` in DeepLIIF's training command : the default value is `./checkpoints/{experiment name}`, where experiment name is defined by `--name` in the same command

If you run the training as is without changing any files, the effective storage volume is `AdditionalDeepLIIFVolume` and the directory there for the training job is `checkpoints/Test_Model_wmla_cpd-wmla-{job id}`. To view or download, you can

- go to Instances -> the storage volume, where you will see a file manager to browse/upload/download files; OR
- in a Watson Studio project where this storage volume is added as data connection, any Jupyter / JupyterLab / Rstudio runtime will auto-mount this volume so you can navigate to `/mnts/AdditionalDeepLIIFVolume/checkpoints/Test_Model_wmla_cpd-wmla-{job id}`

### Use Visdom Visualizer to Track Training Progress

#### 1. Visdom Input Files
Before submitting the training job, make sure you added argument `--remote True` to the DeepLIIF training command in `train_command.py`. This triggers DeepLIIF code to save input to visdom as pickle files and transfer them to the storage volume and directory where your Watson Studio runtime can directly access, similar to saving the trained model/checkpoints mentioned before.

Visdom pickle files are saved in the same location as the model files under `./pickle`. If you run the training as is without changing any files, in a Watson Studio runtime you will find the visdom files at `/mnts/AdditionalDeepLIIFVolume/checkpoints/Test_Model_wmla_cpd-wmla-{job id}/pickle`.

#### 2. Watson Studio Environment
The Watson Studio project needs to have a data connection configured with this storage volume you specified for training. This allows the Watson Studio runtime (Jupyter / JupyterLab / Rstudio) to be able to treat the storage volume as a local directory and get updated content in real time.

Also, the Watson Studio runtime you use should be configured with a special software version (not the default one). In these special software versions, visdom (and other commonly used visualizers) are pre-installed, some patched. The environments are configured to allow local-hosting apps within runtime to be exposed and generate a url to open in browser. As of the time this document is created, these special software versions you may use are:
- Default JupyterLab 3.8 (jupyter-lab-py38-applications)
- Default JupyterLab 3.8 (jupyter-lab-py38gpu-applications)

Monitor the training performance does not take much resource. Even a 1 vCPU 2GB RAM environment can handle it. It is recommended to use a small, non-GPU environment if there is no need to run compute intensive tasks in Watson Studio.

#### 3. Start Visdom for DeepLIIF
After you submit the training job, you will receive the job id and know the directory to visdom files, as explained before. In JupyterLab, run the following command to start visdom:
```
python cli.py visualize --pickle-dir /mnts/{storage volume name}/{checkpoints directory}/pickle
```

If keeping the configurations the same as is, you will have a command like following:
```
python cli.py visualize --pickle-dir /mnts/AdditionalDeepLIIFVolume/checkpoints/Test_Model_wmla_cpd-wmla-227/pickle
```

#### 4. Get Link to Visdom Visualizer
Use the helper functions to find the link to any running application:
```
from ws_applications import display_link
display_link()
```

Or, you can also generate the link to a specific port (Visdom by default uses 8097):
```
from ws_applications import make_link
make_link(8097)
```

## GPU Consumption During Training
The shell script `monitor_gpu.sh` is used to print GPU consumption status regularly in training log, leveraging python package `gpustat`. The training execution script `train_command.py` or `train_command_consumer_local.py` has a line to run this script at back-end:
```
subprocess.Popen('bash monitor_gpu.sh',shell=True)
```
