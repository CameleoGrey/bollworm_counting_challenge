
import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from classes.paths_config import *
from classes.utils import *
from classes.DatasetBuilder_YOLOv5 import DatasetBuilder_YOLOv5
from classes.OptimalBollwormCounter import OptimalBollwormCounter
from PIL import Image
import torch
from torchvision import transforms
import subprocess

from classes.YOLOv5FeatureExtractor import YOLOv5FeatureExtractor


##################################
# eval on cached train_df
"""model_path = os.path.join( models_dir, "model_no_outliers_full_data_100_epochs.pt" )
bollworm_counter = OptimalBollwormCounter( model_path, device="cuda" )


#train_df = pd.read_csv(os.path.join( data_dir, "Train.csv" ))
train_df = pd.read_csv(os.path.join( interim_dir, "clean_train_df.csv" ))"""
"""bollworm_counter.cache_scaled_train_df(train_df, 
                                       raw_image_dir = raw_image_dir,
                                       cache_dir = train_cache_dir,
                                       target_image_size=(1280, 1280))"""

# test raw model without hyperparameters tuning
"""bollworm_counter.eval_on_cached_labeled_df(train_cache_dir, 
                                           verbose_progress=False, 
                                           verbose_metrics=True, 
                                           use_hyperparams_predictor=False)"""

                                 
###########################
# first order (global) optimization      
"""bollworm_counter.first_order_hyperparameters_optimization(train_cache_dir, n_trials=700, random_seed=45)
save(bollworm_counter, os.path.join(models_dir, "first_order_bollworm_counter_1280_clean.pkl"))
bollworm_counter = load( os.path.join(models_dir, "first_order_bollworm_counter_1280_clean.pkl") )
bollworm_counter.eval_on_cached_labeled_df(train_cache_dir, 
                                           verbose_progress=False, 
                                           verbose_metrics=True, 
                                           use_hyperparams_predictor=False)"""
###########################
# second order (local) optimization
#bollworm_counter = load( os.path.join(models_dir, "first_order_bollworm_counter_1280_clean.pkl") )
###############
# change image feature extractor in the cached first order bollworm counter
#bollworm_counter.image_feature_extractor = OptimalBollwormCounter(os.path.join( models_dir, "model_no_outliers_full_data_100_epochs.pt" )).image_feature_extractor
#save(bollworm_counter, os.path.join(models_dir, "first_order_bollworm_counter_1280_clean.pkl"))
###############
#second_order_dataset = bollworm_counter.build_second_order_dataset(train_cache_dir, verbose_progress = True, n_trials = 300, random_seed = 45)
#save(second_order_dataset, os.path.join(models_dir, "second_order_dataset_1280_effnetv2_clean.pkl"))
second_order_dataset = load( os.path.join(models_dir, "second_order_dataset_1280_effnetv2_clean.pkl") )
bollworm_counter = load( os.path.join(models_dir, "first_order_bollworm_counter_1280_clean.pkl") )
bollworm_counter.second_order_hyperparameters_optimization(second_order_dataset, train_cache_dir, random_seed=45)
save(bollworm_counter, os.path.join(models_dir, "second_order_bollworm_counter_1280_clean.pkl"))

"""bollworm_counter = load( os.path.join(models_dir, "second_order_bollworm_counter_1280_clean.pkl") )
bollworm_counter.eval_on_cached_labeled_df(train_cache_dir, 
                                           verbose_progress=False, 
                                           verbose_metrics=True, 
                                           use_hyperparams_predictor=True)"""
##################################

print("done")