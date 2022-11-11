
import numpy as np
from tqdm import tqdm

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from classes.ModelAveragerRegressor import ModelAveragerRegressor

from lightgbm import LGBMRegressor

class HyperparametersPredictor():
    def __init__(self, n_jobs=8, random_seed=45):
        
        self.model = MultiOutputRegressor(estimator=ExtraTreesRegressor(n_estimators=300, 
                                                                        max_depth = 30,
                                                                        n_jobs=n_jobs,
                                                                        random_state = random_seed),
                                          n_jobs=1)
        
        
        pass
    
    def make_correct_dataset(self, second_order_dataset, random_seed=45):
        
        # second_order_dataset = [ image_net_features, target_hyperparameters, target_infos, trial_scores, default_hyperparameters, image_ids ]
        
        #image_ids = second_order_dataset[5]
        #target_infos = second_order_dataset[2]
        
        # clean corrupted train samples
        """dataset_row_ids = np.array( [i for i in range(len(second_order_dataset[0]))] )
        delta_q = np.quantile( target_infos[:, 6], q=0.99 )
        corrupted_train_ids = dataset_row_ids[ (target_infos[:, 0] == 0) & 
                                              (target_infos[:, 1] == 0) & 
                                              (target_infos[:, 6] > delta_q)]
        #corrupted_image_ids = image_ids[ corrupted_train_ids ]
        for ct_id in corrupted_train_ids:
            second_order_dataset[1][ct_id] = second_order_dataset[4].copy()"""
            
        
        return second_order_dataset
    
    def fit(self, second_order_dataset, random_seed=45):
        
        second_order_dataset = self.make_correct_dataset(second_order_dataset, random_seed)
        
        x_train, y_train = second_order_dataset[0], second_order_dataset[1]
        
        ####
        # debug
        #x_train = x_train[:400]
        #y_train = y_train[:400]
        ####
        
        ############
        # train on subsample and eval on validation
        """x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, test_size=0.1, random_state=random_seed )
        partial_models = []
        for i in tqdm(range(y_train.shape[1]), desc="Training partial hyperparams predictor"):
            #partial_model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=8)
            #partial_model.fit( x_train, y_train[:, i], eval_set=(x_val, y_val[:, i]), early_stopping_rounds = 200 )
            partial_model = GreyAverRegressor(n_jobs=8, optimize_weights=True, weights_opt_iters=200,
                                              optimize_hyperparams=False, gbm_opt_iters=100, max_opt_time=2*60*60)
            partial_model.fit( x_train, y_train[:, i], x_val, y_val[:, i] )
            partial_models.append( partial_model )
        self.model.estimators_ = partial_models
        
        y_pred = self.model.predict( x_val )
        
        mae_scores = []
        mape_scores = []
        for i in range(y_pred.shape[1]):
            partial_y_pred = y_pred[:, i]
            partial_y_val = y_val[:, i]
            
            mae_score = mean_absolute_error( partial_y_val, partial_y_pred )
            mae_scores.append( mae_score )
            
            mape_score = mean_absolute_percentage_error( partial_y_val, partial_y_pred )
            mape_scores.append( mape_score )
        print("Validation MAE scores: {}".format(mae_scores))
        print("Validation MAPE scores: {}".format(mape_scores))"""
        #########################################
        
        #########################################
        # train on full data
        partial_models = []
        for i in tqdm(range(y_train.shape[1]), desc="Training partial hyperparams predictor"):
            partial_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, n_jobs=8, random_state = random_seed)
            partial_model.fit( x_train, y_train[:, i], eval_set=(x_train, y_train[:, i]) )
            #partial_model = ModelAveragerRegressor()
            #partial_model.fit(x_train, y_train[:, i], n_models=10, test_size=0.1, random_state=45)
            partial_models.append( partial_model )
        self.model.estimators_ = partial_models
        
            
        y_pred = self.model.predict( x_train )
        
        mae_scores = []
        mape_scores = []
        for i in range(y_pred.shape[1]):
            partial_y_pred = y_pred[:, i]
            partial_y_val = y_train[:, i]
            
            mae_score = mean_absolute_error( partial_y_val, partial_y_pred )
            mae_scores.append( mae_score )
            
            mape_score = mean_absolute_percentage_error( partial_y_val, partial_y_pred )
            mape_scores.append( mape_score )
        print("Train MAE scores: {}".format(mae_scores))
        print("Train MAPE scores: {}".format(mape_scores))
        #########################################
        
        return self
    
    def predict(self, image_features):
        
        image_features = np.array([image_features])
        y_pred = self.model.predict( image_features )
        y_pred = y_pred[0]
        
        return y_pred