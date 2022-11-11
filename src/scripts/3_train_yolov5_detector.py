
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from classes.paths_config import *
from classes.utils import *
import subprocess


#########################################################################
# baseline 640
"""
dataset_path = os.path.join(data_dir, "raw_bollworm_640")
dataset_config_path = os.path.join(dataset_path, "dataset_config.yaml")
image_size = 640
batch_size = 16
epochs = 500
weights = "yolov5s.pt"
cache_type = "ram"
random_seed = 45
save_period = 10
early_stopping_rounds = 20
optimizer = "SGD"
"""
#########################################################################

#########################################################################
# large 640
"""
dataset_path = os.path.join(data_dir, "raw_bollworm_640")
dataset_config_path = os.path.join(dataset_path, "dataset_config.yaml")
image_size = 640
batch_size = 16
epochs = 500
weights = "yolov5l.pt"
cache_type = "ram"
random_seed = 45
save_period = 10
early_stopping_rounds = 4
optimizer = "SGD"
"""
#########################################################################

#########################################################################
# m6 1280
dataset_path = os.path.join(data_dir, "raw_bollworm_1280")
dataset_config_path = os.path.join(dataset_path, "dataset_config.yaml")
image_size = 1280
batch_size = 6
epochs = 500
weights = "yolov5m6.pt"
cache_type = "disk"
random_seed = 45
save_period = 10
early_stopping_rounds = 20
optimizer = "SGD"
#########################################################################

yolov5_train_script_path = "F:/Soft/Anaconda3/envs/py_3_10/Lib/site-packages/yolov5/train.py"

subprocess.call("python {} --img {} --batch-size {} --epochs {} --data {} \
                 --weights {} --cache {} --seed {} \
                 --save-period {} --optimizer {} --patience {} \
                 --workers 1 --noplots".format( yolov5_train_script_path, image_size, batch_size,
                                                epochs, dataset_config_path, weights, cache_type,
                                                random_seed, save_period, optimizer, early_stopping_rounds), shell=False)

print("done")