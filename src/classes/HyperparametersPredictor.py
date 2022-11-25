import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from hashlib import sha256

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from classes.ModelAveragerRegressor import ModelAveragerRegressor
from lightgbm import LGBMRegressor
from yolov5.utils.general import non_max_suppression
import torch
from datetime import datetime

import pandas as pd
from classes.paths_config import *
from classes.utils import *

import optuna

class HyperparametersPredictor():
    def __init__(self, n_jobs=8, random_seed=45):
        
        self.model = MultiOutputRegressor(estimator=ExtraTreesRegressor(n_estimators=300, 
                                                                        max_depth = 30,
                                                                        n_jobs=n_jobs,
                                                                        random_state = random_seed),
                                          n_jobs=1)
        
        
        pass
    
    def make_correct_dataset(self, second_order_dataset, random_seed=45):
        
        # second_order_dataset = [ image_net_features, target_hyperparameters, target_infos, trial_scores, default_hyperparameters, image_ids, raw_predicts ]
        
        image_ids = second_order_dataset[5]
        target_infos = second_order_dataset[2]
        
        # clean corrupted train samples
        dataset_row_ids = np.array( [i for i in range(len(second_order_dataset[0]))] )
        left_delta_q = np.quantile( target_infos[:, 6], q=0.85 )
        #right_delta_q = np.quantile( target_infos[:, 6], q=0.9 )
        corrupted_train_ids = dataset_row_ids[ (target_infos[:, 0] == 0) & 
                                               (target_infos[:, 1] == 0) & 
                                               (target_infos[:, 6] >= left_delta_q) # & 
                                               #(target_infos[:, 6] < right_delta_q)
                                               ]
        corrupted_image_ids = second_order_dataset[5][corrupted_train_ids]
        print(corrupted_image_ids)
        #corrupted_images_df = pd.DataFrame( corrupted_image_ids )
        #corrupted_images_df.to_csv( os.path.join( interim_dir, "corrupted_images_ids.csv" ), index=False )
        
        clean_ids = np.delete( dataset_row_ids, corrupted_train_ids )
        
        for i in [0, 1, 2, 3, 5, 6]:
            second_order_dataset[i] = second_order_dataset[i][clean_ids]
            
        
        return second_order_dataset
    
    def crop_dataset(self, second_order_dataset, sum_delta_q = 0.9, pred_count_q = 0.8):
        
        # second_order_dataset = [ image_net_features, target_hyperparameters, target_infos, trial_scores, default_hyperparameters, image_ids, raw_predicts ]
        
        image_ids = second_order_dataset[5]
        target_infos = second_order_dataset[2]
        
        # clean corrupted train samples
        dataset_row_ids = np.array( [i for i in range(len(second_order_dataset[0]))] )
        delta_unique, delta_counts = np.unique(target_infos[:, 6], return_counts=True)
        pred_unique, pred_counts = np.unique(target_infos[:, 3], return_counts=True)
        sum_delta_threshold = np.quantile( target_infos[:, 6], q = sum_delta_q )
        pred_count_threshold = np.quantile( target_infos[:, 3], q = pred_count_q )
        self.pred_count_threshold = pred_count_threshold
        
        high_error_train_ids = dataset_row_ids[(target_infos[:, 6] >= sum_delta_threshold) & 
                                              (target_infos[:, 3] >= pred_count_threshold)]
        high_error_image_ids = second_order_dataset[5][high_error_train_ids]
        print(high_error_image_ids)
        print(len(high_error_image_ids))
        
        for i in [0, 1, 2, 3, 5, 6]:
            second_order_dataset[i] = second_order_dataset[i][high_error_train_ids]
            
        
        return second_order_dataset
    
    def make_stratification_ids(self, y_train, y_default):
        
        y_default = deepcopy(y_default)
        y_default = str(y_default)
        y_default = y_default.encode()
        y_default = sha256(y_default, usedforsecurity=True)
        y_default = y_default.hexdigest()
        
        y_hashes = []
        for i in range(len(y_train)):
            y_hash = str(y_train[i])
            y_hash = y_hash.encode()
            y_hash = sha256(y_hash, usedforsecurity=True)
            y_hash = y_hash.hexdigest()
            y_hashes.append( y_hash )
        y_hashes = np.array( y_hashes )
        
        stratification_ids = y_hashes == y_default
        stratification_ids = 1 * stratification_ids
        
        
        return stratification_ids
    
    def make_sample_weights(self, y_train, y_default):
        strat_ids = self.make_stratification_ids(y_train, y_default)
        sample_weights = np.zeros((len(strat_ids)), dtype=np.float32)
        uniq_labels, labels_counts = np.unique( strat_ids, return_counts=True )
        
        # N/count
        sample_weights[ strat_ids == 0 ] = len(strat_ids) / labels_counts[0]
        sample_weights[ strat_ids == 1 ] = len(strat_ids) / labels_counts[1]
        
        # 1.0 && 1 / def_count
        #sample_weights[ strat_ids == 0 ] = 1.0
        #sample_weights[ strat_ids == 1 ] =  1 / labels_counts[1]
        
        return sample_weights
        
    def make_mixup_dataset(self, x_train, y_train, sample_weights=None,
                           mixup_size=100_000, alpha=0.2, random_seed=45):
        
        if sample_weights is None:
            sample_weights = np.ones(shape=(len(y_train)), dtype=np.float32)
        
        np.random.seed( random_seed )
        sample_ids = np.linspace(0, len(y_train), num=len(y_train), dtype=np.uint64, endpoint=False )
        
        x_mixup = []
        y_mixup = []
        sample_weight_mixup = []
        for mixup_step in tqdm(range( mixup_size ), desc="building mixup dataset"):
            mix_ids = np.random.choice(sample_ids, size=2, replace=True)
            
            lam = np.random.beta(alpha, alpha)
            x_mix = lam * x_train[mix_ids[0]] + (1.0 - lam) * x_train[mix_ids[1]]
            y_mix = lam * y_train[mix_ids[0]] + (1.0 - lam) * y_train[mix_ids[1]]
            sample_weight_mix = lam * sample_weights[mix_ids[0]] + (1.0 - lam) * sample_weights[mix_ids[1]]
            
            x_mixup.append( x_mix )
            y_mixup.append( y_mix )
            sample_weight_mixup.append( sample_weight_mix )
            
        x_mixup = np.array( x_mixup )
        y_mixup = np.array( y_mixup )
        sample_weight_mixup = np.array( sample_weight_mixup )
        
        return x_mixup, y_mixup, sample_weight_mixup
    
    def make_optimal_sample_weights(self, x_train, y_train, true_class_counts, raw_train_predicts, n_trials=200, random_seed=45):
        
        def extract_class_counts(y_pred, confidence, iou_threshold, max_det):
        
            y_pred = deepcopy(y_pred)
            
            y_pred = non_max_suppression(y_pred, conf_thres=confidence, iou_thres=iou_threshold, 
                                         classes=None, agnostic=False, multi_label=False,
                                         labels=(), max_det=max_det, nm=0)
            y_pred = y_pred[0].cpu().detach().numpy()
            y_pred = y_pred[:, 5].astype(np.int64)
            
            class_names_dict = {0: "abw", 1: "pbw"}
            
            class_counts = {}
            for class_id in class_names_dict.keys():
                class_counts[ class_names_dict[class_id] ] = 0
            
            for y in y_pred:
                class_name = class_names_dict[y]
                class_counts[ class_name ] += 1
                
            return class_counts
        
        def objective(trial):
            
            start_time = datetime.now()
            sample_weights = []
            for i in range(len(x_train)):
                sample_weight = trial.suggest_float( "{}".format(i), 0.0, 2.0 )
                sample_weights.append( sample_weight )
            
            ###################################################
            # for comparison
            #sample_weights = [1.0 for i in range(len(x_train))]
            #sample_weights = self.make_sample_weights(y_train, y_default)
            ###################################################
            
            model = MultiOutputRegressor(estimator=LGBMRegressor(n_estimators=200, learning_rate=0.005, n_jobs=2, random_state = random_seed), n_jobs=6)
            model.fit(x_train, y_train, sample_weight=sample_weights)
            
            predicted_hyper_params = model.predict( x_train )
            
            mae_scores = []
            mape_scores = []
            mse_scores = []
            for i in range(predicted_hyper_params.shape[1]):
                partial_predicted_hyper_params = predicted_hyper_params[:, i]
                partial_y_val = y_train[:, i]
                
                mae_score = mean_absolute_error( partial_y_val, partial_predicted_hyper_params )
                mae_scores.append( mae_score )
            
                mape_score = mean_absolute_percentage_error( partial_y_val, partial_predicted_hyper_params )
                mape_scores.append( mape_score )
                
                mse_score = mean_squared_error( partial_y_val, partial_predicted_hyper_params )
                mse_scores.append( mse_score )
            
            print("Validation MAE scores: {}".format(mae_scores))
            print("Validation MAPE scores: {}".format(mape_scores))
            print("Validation MSE scores: {}".format(mse_scores))
            
            predicted_class_counts = []
            for i in range( len(predicted_hyper_params) ):
                sample_hyper_params = predicted_hyper_params[i]
                abw_parameters = list(sample_hyper_params[:3])
                pbw_parameters = list(sample_hyper_params[3:])
                abw_parameters[2] = int(abw_parameters[2])
                pbw_parameters[2] = int(pbw_parameters[2])
                
                raw_bbox_predict = list(raw_train_predicts[i])
                
                class_count_pred = {}
                class_count_pred["abw"] = extract_class_counts(raw_bbox_predict, *abw_parameters)["abw"]
                class_count_pred["pbw"] = extract_class_counts(raw_bbox_predict, *pbw_parameters)["pbw"]
                
                class_count_pred = np.array([class_count_pred["abw"], class_count_pred["pbw"]])
                predicted_class_counts.append( class_count_pred )
            
            mae_scores = []
            for i in range( len(predicted_class_counts) ):
                y_pred = predicted_class_counts[i]
                y_true = true_class_counts[i]

                abw_delta = np.abs( y_true[0] - y_pred[0] )
                pbw_delta = np.abs( y_true[1] - y_pred[1] )
            
                local_mae = abw_delta + pbw_delta
                mae_scores.append( local_mae )
            mae_score = np.mean( mae_scores )
            
            total_time = datetime.now() - start_time
            trial_i = len( trial.study.trials )
            print("Trial {} MAE: {}".format(trial_i, mae_score))
            print("Trial {} time: {}".format(trial_i, total_time))

            return mae_score
        
        
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        
        # TPE sampler
        study = optuna.create_study(directions=["minimize"], sampler=optuna.samplers.MOTPESampler(seed=random_seed))
        study.optimize( objective, n_trials=n_trials, n_jobs=1 )
        
        scores = []
        best_trials = study.best_trials
        for i in range(len(best_trials)):
            mae = best_trials[i].values[0]
            scores.append(mae)
            
        best_score = scores[np.argmin( scores )]
        best_trial_id = np.argmin( scores )
        best_trial = best_trials[ best_trial_id ]
        best_params = best_trial.params
        
        best_sample_weights = []
        for i in range( len(x_train) ):
            best_sample_weights.append( best_params[str(i)] )
            
        print("Best MAE: {}".format(best_score))
        
        return best_sample_weights
        
    
    def fit(self, second_order_dataset, random_seed=45):
        
        #second_order_dataset = self.make_correct_dataset(second_order_dataset, random_seed)
        second_order_dataset = self.crop_dataset(second_order_dataset)
        
        #########################################
        # train on full data
        
        x_train, y_train = second_order_dataset[0], second_order_dataset[1]
        raw_train_predicts = second_order_dataset[6]
        true_class_counts = second_order_dataset[2][:, :2]
        
        partial_models = []
        #sample_weights = self.make_optimal_sample_weights(x_train, y_train, true_class_counts, raw_train_predicts, n_trials=300, random_seed=45)
        #save(sample_weights, path="./raw_optimal_weights.pkl")
        #sample_weights = self.make_sample_weights(y_train, y_default = second_order_dataset[4])
        x_train, y_train, sample_weights = self.make_mixup_dataset(x_train, y_train, sample_weights=None, 
                                                                   mixup_size = 100_000, alpha=0.2, random_seed=45)
        for i in tqdm(range(y_train.shape[1]), desc="Training partial hyperparams predictor"):
            partial_model = LGBMRegressor(n_estimators=3000, learning_rate=0.05, n_jobs=8, random_state = random_seed)
            partial_model.fit( x_train, y_train[:, i], 
                               sample_weight = sample_weights,
                               eval_set=(x_train, y_train[:, i]) )
            #partial_model = ModelAveragerRegressor()
            #partial_model.fit(x_train, y_train[:, i], n_models=10, test_size=0.1, random_state=45)
            partial_models.append( partial_model )
        self.model.estimators_ = partial_models
        
            
        y_pred = self.model.predict( x_train )
        
        mae_scores = []
        mape_scores = []
        mse_scores = []
        for i in range(y_pred.shape[1]):
            partial_y_pred = y_pred[:, i]
            partial_y_val = y_train[:, i]
            
            mae_score = mean_absolute_error( partial_y_val, partial_y_pred )
            mae_scores.append( mae_score )
            
            mape_score = mean_absolute_percentage_error( partial_y_val, partial_y_pred )
            mape_scores.append( mape_score )
            
            mse_score = mean_squared_error( partial_y_val, partial_y_pred )
            mse_scores.append( mse_score )
            
        print("Validation MAE scores: {}".format(mae_scores))
        print("Validation MAPE scores: {}".format(mape_scores))
        print("Validation MSE scores: {}".format(mse_scores))
        
        print("Validation MAE mean score: {}".format(np.mean(mae_scores)))
        print("Validation MAPE mean score: {}".format(np.mean(mape_scores)))
        print("Validation MSE mean score: {}".format(np.mean(mse_scores)))
        #########################################
        
        
        ##########################################
        # train on subsample and eval on validation
        """x_train, y_train = second_order_dataset[0], second_order_dataset[1]
        
        stratification_ids = self.make_stratification_ids(y_train, y_default = second_order_dataset[4])
        ####
        # debug
        #x_train = x_train[:400]
        #y_train = y_train[:400]
        ####
        
        x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, stratify=stratification_ids, 
                                                           test_size=0.2, random_state=random_seed )
        
        raw_train_predicts = second_order_dataset[6]
        sample_weights = self.make_optimal_sample_weights(x_train, y_train, raw_train_predicts, n_trials=2, random_seed=45)
        
        x_train, y_train, sample_weights = self.make_mixup_dataset(x_train, y_train, sample_weights, 
                                                                   mixup_size = 100_000, alpha=1.0, random_seed=45)
        
        ##########
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit( y_train )
        y_train = scaler.transform( y_train )
        y_val = scaler.transform( y_val )
        ##########
        
        partial_models = []
        for i in tqdm(range(y_train.shape[1]), desc="Training partial hyperparams predictor"):
            partial_model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=8)
            partial_model.fit( x_train, y_train[:, i],
                               sample_weight = sample_weights,
                               eval_set=(x_val, y_val[:, i]), 
                               early_stopping_rounds = 50 )
            
            #partial_model = GreyAverRegressor(n_jobs=8, optimize_weights=True, weights_opt_iters=200,
            #                                  optimize_hyperparams=False, gbm_opt_iters=100, max_opt_time=2*60*60)
            #partial_model.fit( x_train, y_train[:, i], x_val, y_val[:, i] )
            
            partial_models.append( partial_model )
        self.model.estimators_ = partial_models
        
        y_pred = self.model.predict( x_val )
        
        mae_scores = []
        mape_scores = []
        mse_scores = []
        for i in range(y_pred.shape[1]):
            partial_y_pred = y_pred[:, i]
            partial_y_val = y_val[:, i]
            
            mae_score = mean_absolute_error( partial_y_val, partial_y_pred )
            mae_scores.append( mae_score )
            
            mape_score = mean_absolute_percentage_error( partial_y_val, partial_y_pred )
            mape_scores.append( mape_score )
            
            mse_score = mean_squared_error( partial_y_val, partial_y_pred )
            mse_scores.append( mse_score )
            
        print("Validation MAE scores: {}".format(mae_scores))
        print("Validation MAPE scores: {}".format(mape_scores))
        print("Validation MSE scores: {}".format(mse_scores))
        
        print("Validation MAE mean score: {}".format(np.mean(mae_scores)))
        print("Validation MAPE mean score: {}".format(np.mean(mape_scores)))
        print("Validation MSE mean score: {}".format(np.mean(mse_scores)))"""
        #########################################
        
        return self
    
    def predict(self, image_features):
        
        image_features = np.array([image_features])
        y_pred = self.model.predict( image_features )
        y_pred = y_pred[0]
        
        return y_pred