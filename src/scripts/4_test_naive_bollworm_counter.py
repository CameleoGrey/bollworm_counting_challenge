
import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from classes.paths_config import *
from classes.utils import *
from classes.DatasetBuilder_YOLOv5 import DatasetBuilder_YOLOv5
from classes.NaiveBollwormCounter import NaiveBollwormCounter
from PIL import Image
import torch
from torchvision import transforms
import subprocess


####################################
"""yolov5_detect_script_path = "F:/Soft/Anaconda3/envs/py_3_10/Lib/site-packages/yolov5/detect.py"
weights = "F:/MyProgs/zindi_bollworm/models/failure_baseline_640.pt"
image_id = "id_00c5e18f37b15519ace21ce2.jpg"
image_path = os.path.join( raw_image_dir, image_id )

subprocess.call("python {} --source {} --weights {} --conf-thres 0.5".format( yolov5_detect_script_path,
                                                            image_path,
                                                            weights), shell=False)"""
###################################                                                            

###################################
# test predict
"""model_path = "F:/MyProgs/zindi_bollworm/models/0_baseline_640.pt"
bollworm_counter = NaiveBollwormCounter( model_path, device="cuda" )
image_id = "id_1a5aa99bad41a520fd5c80b6.jpg"
image_path = os.path.join( raw_image_dir, image_id )
readed_image = DatasetBuilder_YOLOv5().read_image(raw_image_dir, image_id)

bollworm_counts = bollworm_counter.predict(readed_image, 
                                           confidence=0.25, 
                                           iou_threshold=0.45, 
                                           target_image_size=(640, 640))
print( bollworm_counts )"""
##################################

##################################
# eval on cached train_df
model_path = "F:/MyProgs/zindi_bollworm/models/0_baseline_640.pt"
bollworm_counter = NaiveBollwormCounter( model_path, device="cuda" )
train_cache_dir = os.path.join(data_dir, "train_df_cache")

train_df = pd.read_csv(os.path.join( data_dir, "Train.csv" ))
bollworm_counter.cache_scaled_train_df(train_df, 
                                       raw_image_dir = raw_image_dir,
                                       cache_dir = train_cache_dir,
                                       target_image_size=(640, 640))

bollworm_counter.eval_on_cached_labeled_df(train_cache_dir,
                                           confidence=0.25, 
                                           iou_threshold=0.45, 
                                           max_det=600, 
                                           target_image_size=(640, 640))
##################################

print("done")