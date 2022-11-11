
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.transforms import PILToTensor
from yolov5.utils.general import non_max_suppression

from classes.YOLOv5TorchBackend import YOLOv5TorchBackend
from classes.utils import *

class NaiveBollwormCounter():
    def __init__(self, yolov5_checkpoint_path, device="cuda"):
        
        self.model = YOLOv5TorchBackend(weights=yolov5_checkpoint_path,
                                        device=torch.device(device))
        self.device = device
        self.class_names_dict = self.model.names
        
        pass
    
    def cache_scaled_train_df(self, train_df, cache_dir, raw_image_dir, target_image_size=(640, 640)):
        
        unique_image_ids = np.array(list(set(train_df["image_id_worm"].to_numpy())))
        n_images = len(unique_image_ids)
        for i in tqdm(range(n_images), desc="Caching Train.csv"):
            
            y_true = {"abw": 0, "pbw": 0}
            labeled_subsample = train_df[train_df["image_id_worm"] == unique_image_ids[i]]
            for j in range(len(labeled_subsample)):
                labeled_row = labeled_subsample.iloc[j]
                worm_type = labeled_row["worm_type"]
                if worm_type != str("nan"):
                    y_true[worm_type] = labeled_row["number_of_worms"]
            
            image_id = unique_image_ids[i]
            image_path = os.path.join( raw_image_dir, image_id )
            readed_image = Image.open(image_path)
            preprocessed_image = self.prerpocess_image(readed_image, target_image_size)
            
            sample = [ image_id, preprocessed_image, y_true ] 
            
            save( sample, os.path.join( cache_dir, "{}.pkl".format(image_id) ), verbose=False )

        
        pass
    
    def eval_on_cached_labeled_df(self, cache_dir, 
             confidence=0.25, iou_threshold=0.45, max_det=600, 
             target_image_size=(640, 640) ):
        
        cached_samples_names = os.listdir(cache_dir)
        
        score_mae = 0.0
        n_images = len(cached_samples_names)
        for i in range(n_images):
            
            sample_path = os.path.join( cache_dir, cached_samples_names[i])
            sample = load( sample_path, verbose=False )
            
            image_id = sample[0]
            image = sample[1]
            y_true = sample[2]
            
            try:
                y_pred = self.predict(image, 
                                      preprocess_image=False, 
                                      confidence = confidence, 
                                      iou_threshold = iou_threshold, 
                                      max_det = max_det, 
                                      target_image_size = target_image_size)
            except Exception:
                print("Bad image")
                continue
            
            abw_delta = np.abs( y_true["abw"] - y_pred["abw"] )
            pbw_delta = np.abs( y_true["pbw"] - y_pred["pbw"] )
            
            sum_delta = abw_delta + pbw_delta
            score_mae_i = sum_delta / n_images
            score_mae += score_mae_i
            
            print("Sample: {} of {} | image_id: {} | \
            abw_true {} pbw_true {} \
            abw_pred {} pbw_pred {} \
            abw_delta {} pbw_delta {} sum_delta {}".format(i, n_images, image_id,
                                                           y_true["abw"], y_true["pbw"],
                                                           y_pred["abw"], y_pred["pbw"],
                                                           abw_delta, pbw_delta, sum_delta))
        
        print("MAE: {}".format(score_mae))
            
        return score_mae
    
    def eval_on_raw_labeled_df(self, labeled_df, raw_image_dir, 
             confidence=0.25, iou_threshold=0.45, max_det=600, 
             target_image_size=(640, 640) ):
        
        score_mae = 0.0
        unique_image_ids = np.array(list(set(labeled_df["image_id_worm"].to_numpy())))
        n_images = len(unique_image_ids)
        for i in range(n_images):
            
            y_true = {"abw": 0, "pbw": 0}
            labeled_subsample = labeled_df[labeled_df["image_id_worm"] == unique_image_ids[i]]
            for j in range(len(labeled_subsample)):
                labeled_row = labeled_subsample.iloc[j]
                worm_type = labeled_row["worm_type"]
                if worm_type != str("nan"):
                    y_true[worm_type] = labeled_row["number_of_worms"]
            
            image_id = unique_image_ids[i]
            image_path = os.path.join( raw_image_dir, image_id )
            readed_image = Image.open(image_path)
            try:
                y_pred = self.predict(readed_image, 
                                      preprocess_image=True, 
                                      confidence = confidence, 
                                      iou_threshold = iou_threshold, 
                                      max_det = max_det, 
                                      target_image_size = target_image_size)
            except Exception:
                print("Bad image")
                continue
            
            abw_delta = np.abs( y_true["abw"] - y_pred["abw"] )
            pbw_delta = np.abs( y_true["pbw"] - y_pred["pbw"] )
            
            sum_delta = abw_delta + pbw_delta
            score_mae_i = sum_delta / n_images
            score_mae += score_mae_i
            
            print("Sample: {} of {} | image_id: {} | \
            abw_true {} pbw_true {} \
            abw_pred {} pbw_pred {} \
            abw_delta {} pbw_delta {} sum_delta {}".format(i, n_images, image_id,
                                                           y_true["abw"], y_true["pbw"],
                                                           y_pred["abw"], y_pred["pbw"],
                                                           abw_delta, pbw_delta, sum_delta))
        
        print("MAE: {}".format(score_mae))
            
        return score_mae
    
    def cache_scaled_test_df(self, test_df, cache_dir, raw_image_dir, target_image_size=(640, 640)):
        
        unique_image_ids = list(test_df["image_id_worm"].to_numpy())
        n_images = len(unique_image_ids)
        for i in tqdm(range(n_images), desc="Caching scaled Test.csv"):
            
            image_id = unique_image_ids[i]
            image_path = os.path.join( raw_image_dir, image_id )
            readed_image = Image.open(image_path)
            
            preprocessed_image = self.prerpocess_image(readed_image, target_image_size)
            
            sample = [ image_id, preprocessed_image ] 
            
            save( sample, os.path.join( cache_dir, "{}.pkl".format(image_id) ), verbose=False )

        
        pass
    
    def build_submission_on_cached_test(self, cached_test_dir, 
             confidence=0.25, iou_threshold=0.45, max_det=600,
             target_image_size=(640, 640)):
        
        cached_samples_names = os.listdir(cached_test_dir)
        
        predicts = []
        n_images = len(cached_samples_names)
        for i in tqdm(range(n_images), desc="Building submission on the cached test data"):
            
            sample_path = os.path.join( cached_test_dir, cached_samples_names[i])
            sample = load( sample_path, verbose=False )
            
            image_id = sample[0]
            image = sample[1]
            
            y_pred = self.predict(image, 
                                  preprocess_image = False, 
                                  confidence = confidence, 
                                  iou_threshold = iou_threshold, 
                                  max_det = max_det, 
                                  target_image_size = target_image_size)
            
            splitted_image_id = image_id.split(".")
            predict_abw_row = [splitted_image_id[0] + "_abw", y_pred["abw"]]
            predict_pbw_row = [splitted_image_id[0] + "_pbw", y_pred["pbw"]]
            predicts.append( predict_abw_row )
            predicts.append( predict_pbw_row )
        
        submission_df = pd.DataFrame(data=predicts, columns=["image_id_worm", "number_of_worms"])
            
        return submission_df
    
    def build_submission_on_raw_test(self, test_df, raw_image_dir, 
             confidence=0.25, iou_threshold=0.45, max_det=600,
             target_image_size=(640, 640)):
        
        image_ids = test_df["image_id_worm"].to_numpy()
        
        #######
        # debug
        #image_ids = image_ids[:100]
        #######
        
        n_images = len(image_ids)
        predicts = []
        for i in tqdm(range(n_images), desc="Building submission"):
            
            image_path = os.path.join( raw_image_dir, image_ids[i] )
            readed_image = Image.open(image_path)
            y_pred = self.predict(readed_image, confidence, iou_threshold, max_det, target_image_size)
            
            splitted_image_id = image_ids[i].split(".")
            predict_abw_row = [splitted_image_id[0] + "_abw", y_pred["abw"]]
            predict_pbw_row = [splitted_image_id[0] + "_pbw", y_pred["pbw"]]
            predicts.append( predict_abw_row )
            predicts.append( predict_pbw_row )
        
        submission_df = pd.DataFrame(data=predicts, columns=["image_id_worm", "number_of_worms"])
            
        return submission_df
    
    def prerpocess_image(self, image, target_image_size=(640, 640)):
        image = image.resize(target_image_size, resample = Image.Resampling.BICUBIC )
        image = PILToTensor()(image)
        image = image.float()
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        return image
    
    def predict(self, image, preprocess_image, confidence=0.25, iou_threshold=0.45, max_det=600, target_image_size=(640, 640)):
        
        if preprocess_image:
            image = self.prerpocess_image(image, target_image_size)
        
        if len(image.shape) == 3:
            image = image[None]  # expand for batch dim
        image = image.to(self.device)
        
        y_pred = self.model(image)
        #image = image.cpu().detach()
        
        y_pred = non_max_suppression(y_pred, conf_thres=confidence, iou_thres=iou_threshold, 
                                     classes=None, agnostic=False, multi_label=False,
                                     labels=(), max_det=max_det, nm=0)
        y_pred = y_pred[0].cpu().detach().numpy()
        y_pred = y_pred[:, 5].astype(np.int64)
        
        class_counts = {}
        for class_id in self.class_names_dict.keys():
            class_counts[ self.class_names_dict[class_id] ] = 0
        
        for y in y_pred:
            class_name = self.class_names_dict[y]
            class_counts[ class_name ] += 1
        
        
        return class_counts


