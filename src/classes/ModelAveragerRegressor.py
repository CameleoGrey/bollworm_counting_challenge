
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold, ShuffleSplit
from sklearn.utils import compute_sample_weight
from tqdm import tqdm
import optuna
from sklearn import metrics
from pprint import pprint

class ModelAveragerRegressor():
    def __init__(self):

        # default models
        self.base_models = None


        pass

    def fit(self, x_train, y_train, n_models, test_size=0.05, random_state=45):
        
        self.base_models = []
        
        rs = ShuffleSplit(n_splits=n_models, test_size=test_size, random_state=random_state)
        for id_train, id_val in tqdm(rs.split(x_train)):
            partial_model = LGBMRegressor(n_estimators=10000, learning_rate=0.01, n_jobs=8)
            
            x_t, y_t = x_train[id_train], y_train[id_train]
            x_v, y_v = x_train[id_val], y_train[id_val]
            partial_model.fit( x_t, y_t, eval_set=(x_v, y_v), early_stopping_rounds = 200 )
            #partial_model.fit( x_t, y_t )
            
            self.base_models.append( partial_model )


        return self

    def predict(self, x):

        y_pred_batches = []
        for meta_model in self.base_models:
            pred = meta_model.predict( x )
            y_pred_batches.append( pred )

        y_pred = np.mean( y_pred_batches, axis=0 )
        return y_pred
