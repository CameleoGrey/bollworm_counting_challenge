
import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from classes.paths_config import *
from classes.utils import *
from classes.DatasetBuilder_YOLOv5 import DatasetBuilder_YOLOv5
from classes.OptimalBollwormCounter import OptimalBollwormCounter
from classes.YOLOv5FeatureExtractor import YOLOv5FeatureExtractor
from classes.ImageFeatureExtractor import ImageFeatureExtractor
from PIL import Image
import torch
from torchvision import transforms
import subprocess
from sklearn.manifold import TSNE

#from yolov5.models.yolo import DetectionModel

def extract_cached_images_features( cache_dir, feature_extractor ):
        
    cached_samples_names = os.listdir(cache_dir)
    
    ##########
    # debug
    #cached_samples_names = cached_samples_names[:100]
    ##########
    
    n_images = len(cached_samples_names)
    extracted_features = []
    for i in tqdm(range(n_images), desc="Collecting cached images features"):
        
        sample_path = os.path.join( cache_dir, cached_samples_names[i])
        sample = load( sample_path, verbose=False )
        
        image_id = sample[0]
        image = sample[1]
        
        try:
            image_features = feature_extractor.extract_features(image)
        except Exception as e:
            print(e)
            continue
        
        extracted_features.append( image_features )
    extracted_features = np.vstack( extracted_features ) 
        
    return extracted_features

def extract_raw_images_features( image_dir, feature_extractor ):
        
    cached_samples_names = os.listdir(image_dir)
    
    ##########
    # debug
    #cached_samples_names = cached_samples_names[:100]
    ##########
    
    n_images = len(cached_samples_names)
    extracted_features = []
    for i in tqdm(range(n_images), desc="Collecting cached images features"):
        
        image_path = os.path.join( image_dir, cached_samples_names[i])
        image = Image.open( image_path )
        
        try:
            image_features = feature_extractor.extract_features(image)
        except Exception as e:
            print(e)
            continue
        
        extracted_features.append( image_features )
    extracted_features = np.vstack( extracted_features ) 
        
    return extracted_features


model_path = os.path.join( models_dir, "1_baseline_1280.pt" )
yolov5_feature_extractor = YOLOv5FeatureExtractor( model_path, input_size=(1280, 1280), device="cuda", cutoff=None )
#yolov5_feature_extractor = ImageFeatureExtractor()
#image_features = extract_raw_images_features(raw_image_dir, yolov5_feature_extractor)
image_features = extract_cached_images_features(train_cache_dir, yolov5_feature_extractor)
save( image_features, Path( interim_dir, "cached_train_images_features.pkl" ) )
image_features = extract_cached_images_features(test_cache_dir, yolov5_feature_extractor)
save( image_features, Path( interim_dir, "cached_test_images_features.pkl" ) )

train_image_features = load( Path( interim_dir, "cached_train_images_features.pkl" ) )
test_image_features = load( Path( interim_dir, "cached_test_images_features.pkl" ) )
all_image_features = np.vstack([train_image_features, test_image_features])
compressed_features = TSNE( n_iter=2000, perplexity=100.0, n_jobs = 10, random_state=45, verbose=1 ).fit_transform( all_image_features )
train_size = len( train_image_features )
plt.scatter( compressed_features[ : train_size, 0], compressed_features[ : train_size, 1], s=1, c="tab:blue" )
plt.scatter( compressed_features[train_size : , 0], compressed_features[train_size : , 1], s=1, c="tab:green" )
plt.savefig( Path( plots_dir, "0_yolov5_1280_features_no_cutoff.jpg" ), dpi=1000 )

print("done")
