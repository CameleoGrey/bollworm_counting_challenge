
import os
import pandas as pd
from classes.paths_config import *
from classes.utils import *
from classes.NNRegressor import NNRegressor


if __name__ == "__main__":
    nn_regressor = NNRegressor()
    #train_df = pd.read_csv(os.path.join( data_dir, "Train.csv" ))
    #regressor_dataset = nn_regressor.build_dataset(train_df)
    #save( regressor_dataset, os.path.join(interim_dir, "regressor_dataset.pkl") )
    
    regressor_dataset = load( os.path.join(interim_dir, "regressor_dataset.pkl") )
    image_dir = os.path.join( data_dir, "raw_bollworm_640", "train", "images" )
    nn_regressor.fit(regressor_dataset, 
                     image_dir = image_dir, 
                     batch_size=16, 
                     epochs = 30, 
                     learning_rate = 0.001)
    save( nn_regressor, os.path.join(interim_dir, "nn_regressor.pkl") )
    
    
    nn_regressor = load(os.path.join(interim_dir, "nn_regressor.pkl"))
    test_df = pd.read_csv(os.path.join( data_dir, "Test.csv" ))
    submission_df = nn_regressor.build_submission_on_raw_test(test_df, raw_image_dir, device="cuda")
    submission_path = os.path.join(submissions_dir, "resnet_18_224_30_ep_regression.csv")
    submission_df.to_csv( submission_path, index=False )
    
    print("done")