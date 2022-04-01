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

2. Use notebook `wmla_job_submission` to submit the training. You will receive an application id.
3. Edit the application id in `run_visdom_wmla.py`.
4. Start visdom:
```
python run_visdom_wmla.py
```
