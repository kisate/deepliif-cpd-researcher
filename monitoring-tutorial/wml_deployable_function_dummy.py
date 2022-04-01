import os
import subprocess


os.makedirs('./data',exist_ok=True)
subprocess.run('wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz; mv cifar-10-python.tar.gz data/',shell=True)
subprocess.run('cd data/; tar xvzf cifar-10-python.tar.gz',shell=True)


# https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def score( input_data ):

    payload = input_data.get("input_data")[0] # input_data.get("input_data") is a list
   
    batch_id = payload['batch_id']
    batch_label = payload['batch_label']
    if batch_id in range(1,6):
        d = unpickle(f'data/cifar-10-batches-py/data_batch_{batch_id}')

        img_idx = None
        for i, e in enumerate(d[b'batch_label']):
            if e == batch_label:
                img_idx = i
                break

        if img_idx is None:
            return {"predictions":[{"values":[],
                                    "status":"Batch label not found in this batch id."}]}
        else:
            fn = d[b'filenames'][img_idx].decode()
            label_true = d[b'labels'][img_idx]
            return {"predictions":[{"values":[],
                                    "status":"Found.",
                                    "true_label":label_true,
                                    "filename":fn}]}
    else:
        return {"predictions":[{"values":[],
                                "status":"Batch id does not exist. Use a number between 1 and 5."}]}
