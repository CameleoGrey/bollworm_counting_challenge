
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from classes.paths_config import *
from classes.utils import *
from classes.DatasetBuilder_YOLOv5 import DatasetBuilder_YOLOv5
#from classes.NaiveBollwormCounter import NaiveBollwormCounter
from classes.OptimalBollwormCounter import OptimalBollwormCounter
from PIL import Image
import torch
from torchvision import transforms
import subprocess


########################################
# 
"""model_path = os.path.join( models_dir, "0_baseline_640.pt" )
bollworm_counter = NaiveBollwormCounter( model_path, device="cuda" )
test_df = pd.read_csv(os.path.join( data_dir, "Test.csv" ))
submission_df  = bollworm_counter.build_submission_on_raw_test(test_df, raw_image_dir,
                                                   confidence=0.25, iou_threshold=0.45, max_det=300, 
                                                   target_image_size=(640, 640))
submission_path = os.path.join(submissions_dir, "0_baseline_640.csv")
submission_df.to_csv( submission_path, index=False )"""
########################################


########################################
"""model_path = os.path.join( models_dir, "1_baseline_1280.pt" )
bollworm_counter = OptimalBollwormCounter( model_path, device="cuda" )
test_df = pd.read_csv(os.path.join( data_dir, "Test.csv" ))

bollworm_counter.cache_scaled_test_df(test_df, test_cache_dir, raw_image_dir, target_image_size=(1280, 1280))

submission_df = bollworm_counter.build_submission_on_cached_test(test_cache_dir, 
                                                                 use_hyperparams_predictor=False, 
                                                                 verbose_progress=True, 
                                                                 verbose_metrics=True)
submission_path = os.path.join(submissions_dir, "1_first_order_optimization_reverted_weights_1280.csv")
submission_df.to_csv( submission_path, index=False )"""

########################################

########################################
"""bollworm_counter = bollworm_counter = load( os.path.join(models_dir, "first_order_bollworm_counter_1280_clean.pkl") )
test_df = pd.read_csv(os.path.join( data_dir, "Test.csv" ))

#bollworm_counter.cache_scaled_test_df(test_df, test_cache_dir, raw_image_dir, target_image_size=(1280, 1280))

submission_df = bollworm_counter.build_submission_on_cached_test(test_cache_dir, 
                                                                 use_hyperparams_predictor=False, 
                                                                 verbose_progress=True, 
                                                                 verbose_metrics=True)
submission_path = os.path.join(submissions_dir, "1_first_order_optimization_1280_clean.csv")
submission_df.to_csv( submission_path, index=False )"""
########################################

########################################
bollworm_counter = load( os.path.join(models_dir, "second_order_bollworm_counter_1280_clean.pkl") )
test_df = pd.read_csv(os.path.join( data_dir, "Test.csv" ))

#bollworm_counter.cache_scaled_test_df(test_df, test_cache_dir, raw_image_dir, target_image_size=(1280, 1280))

submission_df = bollworm_counter.build_submission_on_cached_test(test_cache_dir, 
                                                                 use_hyperparams_predictor=True, 
                                                                 verbose_progress=True, 
                                                                 verbose_metrics=True)
submission_path = os.path.join(submissions_dir, "1_second_order_optimization_1280_weighted_mixup_02_3000_clean_partial_adaptive.csv")
submission_df.to_csv( submission_path, index=False )
########################################

print("done")