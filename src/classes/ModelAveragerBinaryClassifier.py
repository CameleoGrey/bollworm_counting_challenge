
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_sample_weight
from tqdm import tqdm
import optuna
from sklearn import metrics
from pprint import pprint
from sklearn.metrics import accuracy_score, f1_score

class ModelAveragerBinaryClassifier():
    def __init__(self):

        # default models
        self.base_models = None


        pass

    def fit(self, x_train, y_train, n_models, random_state=45):
        
        self.base_models = []
        
        #shuffle_ids = np.random.shuffle( [i for i in range(len(y_train))] )
        #x_train = x_train[shuffle_ids]
        #y_train = y_train[shuffle_ids]
        
        rs = StratifiedKFold(n_splits=n_models, shuffle=True, random_state=random_state)
        
        acc_scores = []
        f1_scores = []
        i_fold = 1
        for id_train, id_val in tqdm(rs.split(x_train, y_train)):
            partial_model = LGBMClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=8)
            
            x_t, y_t = x_train[id_train], y_train[id_train]
            x_v, y_v = x_train[id_val], y_train[id_val]
            partial_model.fit( x_t, y_t, eval_set=(x_v, y_v), early_stopping_rounds = 200 )
            
            self.base_models.append( partial_model )
            
            y_p = partial_model.predict( x_v )
            acc = accuracy_score(y_v, y_p)
            f1 = f1_score(y_v, y_p)
            
            print("Validation {}-fold Acc score: {}".format(i_fold, acc))
            print("Validation {}-fold F1 scores: {}".format(i_fold, f1))
            acc_scores.append( acc )
            f1_scores.append( f1 )
            i_fold += 1
        
        mean_acc = np.mean( acc_scores )
        mean_f1 = np.mean( f1_scores )
        print("Validation mean Acc score: {}".format(mean_acc))
        print("Validation mean F1 scores: {}".format(mean_f1))

        return self

    def predict(self, x, threshold=0.5):

        y_pred_batches = []
        for meta_model in self.base_models:
            pred = meta_model.predict_proba( x )[0][1]
            y_pred_batches.append( pred )
        
        y_pred_batches = np.array( y_pred_batches )
        
        mean_probas = np.mean( y_pred_batches )
        
        y_pred = 1 if mean_probas >= threshold else 0

        return y_pred
