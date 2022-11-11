
import numpy as np
from tqdm import tqdm

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from classes.ModelAveragerBinaryClassifier import ModelAveragerBinaryClassifier

from lightgbm import LGBMClassifier

from copy import deepcopy

class BollwormTrigger():
    def __init__(self):
        
        self.model = LGBMClassifier()
        
        pass
    
    def fit(self, second_order_dataset, random_seed=45):
        
        x_train, y_train = deepcopy(second_order_dataset[0]), deepcopy(second_order_dataset[2])
        
        y_train = y_train[:, 0:2]
        y_train = np.sum( y_train, axis=1 )
        y_train[y_train >= 1.0] = 1
        y_train[y_train < 1.0] = 0
        
        ####
        # debug
        #x_train = x_train[:400]
        #y_train = y_train[:400]
        ####
        
        ##############
        self.model = ModelAveragerBinaryClassifier()
        self.model.fit(x_train, y_train, n_models=10, random_state=random_seed)
        
        ##############
        
        return self
    
    def predict(self, image_features):
        
        image_features = np.array([image_features])
        y_pred = self.model.predict( image_features )
        #y_pred = y_pred[0]
        
        y_pred = True if y_pred == 1 else False
        
        return y_pred
    