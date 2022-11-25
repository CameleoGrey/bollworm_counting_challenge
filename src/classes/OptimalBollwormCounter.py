
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.io import read_image
from torchvision.transforms import PILToTensor
from yolov5.utils.general import non_max_suppression

from classes.YOLOv5TorchBackend import YOLOv5TorchBackend
from classes.ImageFeatureExtractor import ImageFeatureExtractor
from classes.HyperparametersPredictor import HyperparametersPredictor
from classes.utils import *
from classes.BollwormTrigger import BollwormTrigger
from classes.NNHPPredictor import NNHPPredictor

import optuna

class OptimalBollwormCounter():
    def __init__(self, yolov5_checkpoint_path, device="cuda"):
        
        self.model = YOLOv5TorchBackend(weights=yolov5_checkpoint_path,
                                        device=torch.device(device))
        self.device = device
        self.class_names_dict = self.model.names
        
        
        # first order parameters
        # Note: these are stub global parameters, through first order optimization
        # they will be changed to optimal for a specific model
        # MAE on Train: 2.043031734620566 for baseline_640 small
        # MAE on Train: 1.8932936222656396 for baseline_1280 medium
        """self.confidence_abw = 0.6601263387097805
        self.confidence_pbw = 0.3864052508358511
        self.iou_threshold_abw = 0.3977171502747005
        self.iou_threshold_pbw = 0.297381323529126
        self.max_det_abw = 392
        self.max_det_pbw = 574"""
        
        # "outside box" parameters
        # Note: these are stub global parameters, through first order optimization
        # they will be changed to optimal for a specific model
        # MAE on Train: 3.08 for baseline_640 small
        # MAE on Train: MAE: 2.534969703193999 for baseline_1280 medium
        self.confidence_abw = 0.25
        self.confidence_pbw = 0.25
        self.iou_threshold_abw = 0.45
        self.iou_threshold_pbw = 0.45
        self.max_det_abw = 300
        self.max_det_pbw = 300
        
        
        # second order class members
        self.image_feature_extractor = ImageFeatureExtractor( device=self.device )
        self.hyper_parameters_predictor = None
        #self.bollworm_trigger = None
        
        pass
    
    def first_order_hyperparameters_optimization(self, train_cache_dir, n_trials=200, random_seed=45):
        
        optuna.logging.set_verbosity(optuna.logging.INFO)
        
        raw_predicts, y_trues = self.collect_raw_predicts_and_trues_(train_cache_dir, 
                                                                     verbose_progress=True, 
                                                                     verbose_metrics=False, 
                                                                     use_hyperparams_predictor=False)
        
        
        def objective(trial):
            
            confidence_abw = trial.suggest_float( "confidence_abw", 0.05, 0.95 )
            confidence_pbw = trial.suggest_float( "confidence_pbw", 0.05, 0.95 )
            iou_threshold_abw = trial.suggest_float( "iou_threshold_abw", 0.05, 0.95 )
            iou_threshold_pbw = trial.suggest_float( "iou_threshold_pbw", 0.05, 0.95 )
            max_det_abw = trial.suggest_int( "max_det_abw", 5, 1000 )
            max_det_pbw = trial.suggest_int( "max_det_pbw", 5, 1000 )
            

            y_preds = self.extract_class_counts_(raw_predicts, 
                                                confidence_abw, iou_threshold_abw, max_det_abw, 
                                                confidence_pbw, iou_threshold_pbw, max_det_pbw)
            mae = self.get_mae_(y_trues, y_preds)

            return mae

        # TPE sampler
        study = optuna.create_study(directions=["minimize"], sampler=optuna.samplers.MOTPESampler(seed=random_seed))
        study.optimize( objective, n_trials=n_trials, n_jobs=1 )

        best_trials = study.best_trials
        scores = []
        for i in range(len(best_trials)):
            mae = best_trials[i].values[0]
            scores.append(mae)
            
        best_score = scores[np.argmin( scores )]
        best_trial_id = np.argmin( scores )
        best_trial = best_trials[ best_trial_id ]
        best_params = best_trial.params
        
        
        self.set_hyperparameters(best_params["confidence_abw"], 
                                 best_params["iou_threshold_abw"], 
                                 best_params["max_det_abw"],
                                 best_params["confidence_pbw"], 
                                 best_params["iou_threshold_pbw"], 
                                 best_params["max_det_pbw"])
        
        print("Best MAE: {}".format( best_score ))
        
        print("Best confidence_abw: {}".format( self.confidence_abw ))
        print("Best iou_threshold_abw: {}".format( self.iou_threshold_abw ))
        print("Best max_det_abw: {}".format( self.max_det_abw ))
        
        print("Best confidence_pbw: {}".format( self.confidence_pbw ))
        print("Best iou_threshold_pbw: {}".format( self.iou_threshold_pbw ))
        print("Best max_det_pbw: {}".format( self.max_det_pbw ))

        return self
    
    def collect_raw_predicts_and_trues_(self, cache_dir, verbose_progress=True, verbose_metrics=True, use_hyperparams_predictor=False):
        cached_samples_names = os.listdir(cache_dir)
        
        ##########
        # debug
        #cached_samples_names = cached_samples_names[:100]
        ##########
        
        score_mae = 0.0
        n_images = len(cached_samples_names)
        
        progress_bar = None
        if verbose_progress:
            progress_bar = tqdm(range(n_images), desc="Collecting raw predicts")
        else:
            progress_bar = range(n_images)
        
        raw_predicts = []
        y_trues = []
        for i in progress_bar:
            
            sample_path = os.path.join( cache_dir, cached_samples_names[i])
            sample = load( sample_path, verbose=False )
            
            image_id = sample[0]
            image = sample[1]
            y_true = sample[2]
            
            try:
                y_pred = self.get_raw_predict_(image,preprocess_image = False,
                                               target_image_size = None)
            except Exception as e:
                print(e)
                continue
            
            raw_predicts.append( y_pred )
            y_trues.append( y_true )
        
        return raw_predicts, y_trues
        
    
    def get_raw_predict_(self, image, preprocess_image, target_image_size=(640, 640)):
        
        if preprocess_image:
            image = self.prerpocess_image(image, target_image_size)
        
        if len(image.shape) == 3:
            image = image[None]  # expand for batch dim
        image = image.to(self.device)
        
        y_pred = self.model(image)
        
        for i in range(len(y_pred)):
            if isinstance( y_pred[i], list ):
                for j in range(len(y_pred[i])):
                    y_pred[i][j] = y_pred[i][j].cpu().detach().to(torch.float16)
            else:
                y_pred[i] = y_pred[i].cpu().detach().to(torch.float16)
        
        return y_pred
    
    def extract_class_counts_(self, y_preds, 
                              confidence_abw, iou_threshold_abw, max_det_abw, 
                              confidence_pbw, iou_threshold_pbw, max_det_pbw):
        
        class_counts = []
        for y_pred in y_preds:
        
            local_class_counts = {}
        
            abw_parameters = [confidence_abw, iou_threshold_abw, max_det_abw,]
            pbw_parameters = [confidence_pbw, iou_threshold_pbw, max_det_pbw]
        
            local_class_counts["abw"] = self.extract_class_counts(y_pred, *abw_parameters)["abw"]
            local_class_counts["pbw"] = self.extract_class_counts(y_pred, *pbw_parameters)["pbw"]
            
            class_counts.append( local_class_counts )
        
        return class_counts
    
    def get_mae_(self, y_trues, y_preds):
        
        score_mae = 0.0
        n_images = len(y_trues)
        for y_true, y_pred in zip( y_trues, y_preds ):
        
            abw_delta = np.abs( y_true["abw"] - y_pred["abw"] )
            pbw_delta = np.abs( y_true["pbw"] - y_pred["pbw"] )
            
            sum_delta = abw_delta + pbw_delta
            score_mae_i = sum_delta / n_images
            score_mae += score_mae_i
        return score_mae
    
    def set_hyperparameters(self, 
                            confidence_abw, iou_threshold_abw, max_det_abw,
                            confidence_pbw, iou_threshold_pbw, max_det_pbw):
        
        self.confidence_abw = confidence_abw
        self.confidence_pbw = confidence_pbw
        self.iou_threshold_abw = iou_threshold_abw
        self.iou_threshold_pbw = iou_threshold_pbw
        self.max_det_abw = max_det_abw
        self.max_det_pbw = max_det_pbw
        
        pass
    
    
    def build_second_order_dataset(self, cache_dir, verbose_progress=True, n_trials=200, random_seed=45 ):
        
        cached_samples_names = os.listdir(cache_dir)
        
        ##########
        # debug
        #cached_samples_names = cached_samples_names[:100]
        ##########
        
        n_images = len(cached_samples_names)
        
        progress_bar = None
        if verbose_progress:
            progress_bar = tqdm(range(n_images), desc="Building second order optimization dataset")
        else:
            progress_bar = range(n_images)
        
        image_net_features = []
        target_hyperparameters = []
        target_infos = []
        trial_scores = []
        image_ids = []
        raw_predicts = []
        default_hyperparameters = [self.confidence_abw, self.iou_threshold_abw, self.max_det_abw,
                                   self.confidence_pbw, self.iou_threshold_pbw, self.max_det_pbw]
        
        for i in progress_bar:
            
            self.set_hyperparameters(*default_hyperparameters)
            
            sample_path = os.path.join( cache_dir, cached_samples_names[i])
            sample = load( sample_path, verbose=False )
            
            image_id = sample[0]
            image = sample[1]
            y_true = sample[2]
            
            try:
                # working on image preprocessed for yolov5 to avoid double reading
                situation_features = self.image_feature_extractor.extract_features( image )
                y_pred = self.predict(image,
                                      preprocess_image=False,
                                      target_image_size = None, 
                                      use_hyperparams_predictor=False)
                raw_pred = self.get_raw_predict_(image, preprocess_image=False, target_image_size=None)
            except Exception as e:
                print(e)
                continue
            
            raw_predicts.append( raw_pred )
            
            abw_delta = np.abs( y_true["abw"] - y_pred["abw"] )
            pbw_delta = np.abs( y_true["pbw"] - y_pred["pbw"] )
            
            sum_delta = abw_delta + pbw_delta
            
            target_info = [ y_true["abw"], y_true["pbw"], y_pred["abw"], y_pred["pbw"], abw_delta, pbw_delta, sum_delta ]
            target_infos.append( target_info )
            
            if sum_delta == 0:
                target_hyperparameters.append( default_hyperparameters )
                trial_scores.append(-1.0)
            else:
                situation_optimal_hyperparameters, median_trial_score = self.find_local_optimal_parameters( image, 
                                                                                        y_true, 
                                                                                        n_trials=n_trials, 
                                                                                        random_seed=random_seed )
                target_hyperparameters.append( situation_optimal_hyperparameters )
                trial_scores.append( median_trial_score )
            
            image_ids.append( image_id )
            image_net_features.append( situation_features )
                
            
        self.set_hyperparameters(*default_hyperparameters)
        
        image_net_features = np.array( image_net_features )
        target_hyperparameters = np.array( target_hyperparameters )
        target_infos = np.array( target_infos )
        default_hyperparameters = np.array( default_hyperparameters )
        trial_scores = np.array( trial_scores )
        image_ids = np.array(image_ids)
        raw_predicts = np.array(raw_predicts)
        second_order_dataset = [ image_net_features, target_hyperparameters, target_infos, trial_scores, default_hyperparameters, image_ids, raw_predicts ]
                
        return second_order_dataset
    
    def find_local_optimal_parameters( self, yolov5_image, y_true, n_trials=200, random_seed=45 ):
        
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        
            
        raw_y_pred = self.get_raw_predict_(yolov5_image, preprocess_image = False, target_image_size = None)

        
        def objective(trial):
            
            confidence_abw = trial.suggest_float( "confidence_abw", 0.0, 1.0 )
            confidence_pbw = trial.suggest_float( "confidence_pbw", 0.0, 1.0 )
            iou_threshold_abw = trial.suggest_float( "iou_threshold_abw", 0.0, 1.0 )
            iou_threshold_pbw = trial.suggest_float( "iou_threshold_pbw", 0.0, 1.0 )
            max_det_abw = trial.suggest_int( "max_det_abw", 1, 1000 )
            max_det_pbw = trial.suggest_int( "max_det_pbw", 1, 1000 )
            
            
            y_pred = self.extract_class_counts_([raw_y_pred], 
                                                confidence_abw, iou_threshold_abw, max_det_abw, 
                                                confidence_pbw, iou_threshold_pbw, max_det_pbw)
            y_pred = y_pred[0]

            abw_delta = np.abs( y_true["abw"] - y_pred["abw"] )
            pbw_delta = np.abs( y_true["pbw"] - y_pred["pbw"] )
            
            sum_delta = abw_delta + pbw_delta

            return sum_delta

        # TPE sampler
        study = optuna.create_study(directions=["minimize"], sampler=optuna.samplers.MOTPESampler(seed=random_seed))
        study.optimize( objective, n_trials=n_trials, n_jobs=1 )

        best_trials = study.best_trials
        scores = []
        for i in range(len(best_trials)):
            sum_delta = best_trials[i].values[0]
            scores.append(sum_delta)
            
        
        ##########################################
        # get median of local optimal parameters
        
        # taking last 33% (stabilized) best params
        #stabilized_trials_count =  int(0.33 * len(best_trials)) + 1
        #stabilized_best_trials = best_trials[-stabilized_trials_count : ]
        #best_trials = stabilized_best_trials
        
        best_params = []
        trial_scores = []
        parameter_names = ["confidence_abw", "iou_threshold_abw", "max_det_abw",
                           "confidence_pbw", "iou_threshold_pbw", "max_det_pbw"]
        for best_trial in best_trials:
            local_best = []
            for parameter_name in parameter_names:
                local_best.append( best_trial.params[parameter_name] )
            best_params.append( local_best )
            trial_scores.append( best_trial.values[0] )
        situation_optimal_parameters = np.median( best_params, axis=0 )
        median_trial_score = np.median( trial_scores )

        #########################################
        
        return situation_optimal_parameters, median_trial_score
    
    def second_order_hyperparameters_optimization(self, second_order_dataset, train_cache_dir, n_jobs=8, random_seed=45 ):
        
        #self.bollworm_trigger = BollwormTrigger()
        #self.bollworm_trigger.fit( second_order_dataset )
        
        self.hyper_parameters_predictor = HyperparametersPredictor(n_jobs=n_jobs, 
                                                                   random_seed=random_seed)
        self.hyper_parameters_predictor.fit(second_order_dataset)
        
        #self.hyper_parameters_predictor = NNHPPredictor()
        #self.hyper_parameters_predictor.fit(second_order_dataset, train_cache_dir, batch_size=8, epochs = 50, learning_rate = 0.0001)
        
        
        return self
    
    def prerpocess_image(self, image, target_image_size=(640, 640)):
        image = image.resize(target_image_size, resample = Image.Resampling.BICUBIC )
        image = PILToTensor()(image)
        image = image.float()
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        return image
    
    
    def predict(self, image, preprocess_image, target_image_size=(640, 640), use_hyperparams_predictor=False):
        
        self.model.eval()
        
        abw_parameters = [self.confidence_abw, self.iou_threshold_abw, self.max_det_abw]
        pbw_parameters = [self.confidence_pbw, self.iou_threshold_pbw, self.max_det_pbw]
        
        if preprocess_image:
            image = self.prerpocess_image(image, target_image_size)
        
        if use_hyperparams_predictor:
            situation_features = self.image_feature_extractor.extract_features(image)
            
            #bollworm_detected = self.bollworm_trigger.predict( situation_features )
            #if not bollworm_detected:
            #    class_counts = {}
            #    class_counts["abw"] = 0
            #    class_counts["pbw"] = 0
            #    return class_counts
            
            situation_hyperparams = self.hyper_parameters_predictor.predict(situation_features)
            
            #situation_hyperparams = self.hyper_parameters_predictor.predict(image)
            
            
            abw_parameters = situation_hyperparams[:3]
            pbw_parameters = situation_hyperparams[3:]
        
        if len(image.shape) == 3:
            image = image[None]  # expand for batch dim
        image = image.to(self.device)
        
        y_pred = self.model(image)
        class_counts = {}
        
        abw_parameters = list(abw_parameters)
        pbw_parameters = list(pbw_parameters)
        abw_parameters[2] = int(abw_parameters[2])
        pbw_parameters[2] = int(pbw_parameters[2])
        
        class_counts["abw"] = self.extract_class_counts(y_pred, *abw_parameters)["abw"]
        class_counts["pbw"] = self.extract_class_counts(y_pred, *pbw_parameters)["pbw"]
        
        return class_counts
    
    def extract_class_counts(self, y_pred, confidence, iou_threshold, max_det):
        
        y_pred = deepcopy(y_pred)
        
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
    
    def eval_on_cached_labeled_df(self, cache_dir, verbose_progress=True, verbose_metrics=True, use_hyperparams_predictor=False ):
        
        cached_samples_names = os.listdir(cache_dir)
        
        ##########
        # debug
        #cached_samples_names = cached_samples_names[:100]
        ##########
        
        score_mae = 0.0
        n_images = len(cached_samples_names)
        
        progress_bar = None
        if verbose_progress:
            progress_bar = tqdm(range(n_images), desc="Evaluating on cached Train.csv")
        else:
            progress_bar = range(n_images)
        
        for i in progress_bar:
            
            sample_path = os.path.join( cache_dir, cached_samples_names[i])
            sample = load( sample_path, verbose=False )
            
            image_id = sample[0]
            image = sample[1]
            y_true = sample[2]
            
            try:
                y_pred = self.predict(image,preprocess_image = False,
                                      target_image_size = None, 
                                      use_hyperparams_predictor = use_hyperparams_predictor)
            except Exception as e:
                print(e)
                continue
            
            abw_delta = np.abs( y_true["abw"] - y_pred["abw"] )
            pbw_delta = np.abs( y_true["pbw"] - y_pred["pbw"] )
            
            sum_delta = abw_delta + pbw_delta
            score_mae_i = sum_delta / n_images
            score_mae += score_mae_i
            
            if verbose_metrics:
                print("Sample: {} of {} | image_id: {} | \
                abw_true {} pbw_true {} \
                abw_pred {} pbw_pred {} \
                abw_delta {} pbw_delta {} sum_delta {}".format(i, n_images, image_id,
                                                               y_true["abw"], y_true["pbw"],
                                                               y_pred["abw"], y_pred["pbw"],
                                                               abw_delta, pbw_delta, sum_delta))
        
        if verbose_metrics:
            print("MAE: {}".format(score_mae))
            
        return score_mae
    
    
    def build_submission_on_cached_test(self, cached_test_dir, 
                                        use_hyperparams_predictor, 
                                        target_image_size=None, 
                                        verbose_progress=False, 
                                        verbose_metrics=True):
        
        cached_samples_names = os.listdir(cached_test_dir)
        
        predicts = []
        n_images = len(cached_samples_names)
        
        progress_bar = None
        if verbose_progress:
            progress_bar = tqdm(range(n_images), desc="Building submission on the cached test data")
        else:
            progress_bar = range(n_images)
        
        for i in progress_bar:
            
            sample_path = os.path.join( cached_test_dir, cached_samples_names[i])
            sample = load( sample_path, verbose=False )
            
            image_id = sample[0]
            image = sample[1]
            
            y_pred = self.predict(image, 
                                  preprocess_image = False, 
                                  target_image_size = target_image_size,
                                  use_hyperparams_predictor = use_hyperparams_predictor)
            
            splitted_image_id = image_id.split(".")
            predict_abw_row = [splitted_image_id[0] + "_abw", y_pred["abw"]]
            predict_pbw_row = [splitted_image_id[0] + "_pbw", y_pred["pbw"]]
            predicts.append( predict_abw_row )
            predicts.append( predict_pbw_row )
            
            if verbose_metrics:
                print("Sample: {} of {} | image_id: {} | abw_pred {} pbw_pred {}".format(i, n_images, image_id, y_pred["abw"], y_pred["pbw"]))
        
        submission_df = pd.DataFrame(data=predicts, columns=["image_id_worm", "number_of_worms"])
            
        return submission_df
    
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
    
    def eval_on_raw_labeled_df(self, labeled_df, raw_image_dir, 
             confidence=0.25, iou_threshold=0.45, max_det=300, 
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
    
    def build_submission_on_raw_test(self, test_df, raw_image_dir, 
             confidence=0.25, iou_threshold=0.45, max_det=300,
             target_image_size=(640, 640)):
        
        image_ids = test_df["image_id_worm"].to_numpy()
        
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
    


