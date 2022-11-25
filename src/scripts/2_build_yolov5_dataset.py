
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from classes.paths_config import *
from classes.utils import *
from classes.DatasetBuilder_YOLOv5 import DatasetBuilder_YOLOv5

dataset_builder = DatasetBuilder_YOLOv5()

bbox_df = pd.read_csv(os.path.join( data_dir, "images_bboxes.csv" ))
#train_df = pd.read_csv(os.path.join( data_dir, "Train.csv" ))
train_df = pd.read_csv(os.path.join( interim_dir, "clean_train_df.csv" ))


dataset_builder.build_yolov5_dataset(raw_images_dir = raw_image_dir,
                                       parent_build_dir = data_dir, 
                                       dataset_build_name = "clean_bollworm_1280", 
                                       bbox_df = bbox_df, 
                                       train_df = train_df,
                                       val_part = 0.05, 
                                       random_seed = 45,
                                       target_image_size = (1280, 1280))

print("done")