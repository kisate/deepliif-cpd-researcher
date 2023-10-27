# DL Training Tutorial in CPD (PyTorch)

This is a tutorial modified based on [visdom-tutorial](https://github.com/noagarcia/visdom-tutorial) that showcased how to train a pytorch model in Watson Studio JupyterLab environment (develop the code and test), and how to modify the resulting scripts to run on Watson Machine Learning Accelerator (WMLA).

## WS
- train.py
- utils.py
- run_visdom.py

#### How to

Start visdom:
```
python run_visdom.py
```

Start training:
```
python train.py
```

## WMLA
- train_wmla.py
- utils_wmla.py
- run_visdom_wmla.py

To transfer checkpoints (visdom data, trained model, etc.) back to storage volume, it has dependency on the following util scripts:
- storage_volume_utils.py
- cpd_utils.py

#### How to:
1. Make sure you have added a storage volume as data connection to the project, and your compute environment was started after the data connection is created. If this is the case, you should be able to see a folder for your storage volume:
```
ls -lh /mnts
```
Edit the volume name in `run_visdom_wmla.py` using this folder name.

Also update `VOLUME_DISPLAY_NAME` in `wmla_job_submission` notebook with the name of the volume.

2. Use notebook `wmla_job_submission` to submit the training. You will receive an application id.
3. Edit the application id in `run_visdom_wmla.py`.
4. Start visdom:
```
python run_visdom_wmla.py
```

## Additional Packages
Watson Studio runtime and Watson Machine Learning Accelerator runtime are different, and they do not share the same configuration. This means you need to configure both if you need to bring in additional library, such as `visdom`:
- In WS, you can add it into the Customization section in your environment configuration before starting one.
- In WMLA, there is no UI to control this, and you will need a line or two in your script to install extra packages, for example
```
import subprocess
subprocess.run("pip install visdom --user",shell=True)
```

#### Different behavior of package installation in WS vs. in WMLA

In WS, the additional packages installed via code within a runtime or via the Customization section in environment configuration are essentially **temporary**. Every time the a new environment / pod starts, the same installation process needs / will be executed again. The installed packages won't be memorized.

Contrary to WS, in WMLA the packages installed via code in a training script are saved permanently. The next time you submit the same script, you shall see messages indicating "Requirement already satisfied" even though it is a new environment / pod.