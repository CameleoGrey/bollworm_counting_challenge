
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from datetime import datetime

from classes.utils import *
from classes.ImageFeatureExtractor import ImageFeatureExtractor 

from lightgbm import LGBMRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import AutoAugment, Resize, Compose, ConvertImageDtype, PILToTensor, Normalize, InterpolationMode
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

class TrainDataset(Dataset):
    def __init__(self, regressor_dataset, image_dir, device):
        #self.data = data
        #self.label = label
        
        self.image_ids = regressor_dataset[0]
        self.targets = regressor_dataset[1]
        self.images_dir = image_dir
        self.device = device
        
        #######
        # debug
        #self.image_ids = self.image_ids[:100]
        #######
        

    def __getitem__(self, index):
            
        image_id = self.image_ids[index]
        image_path = os.path.join( self.images_dir, image_id )
        
        image = read_image(image_path, mode=ImageReadMode.RGB)
        y_true = self.targets[index]
        
        y = y_true
        #y = np.log1p( y_true )
        y = torch.Tensor(y).to(torch.float32)
        y = y.to(self.device)
        
        x = image.to( self.device )
        x = Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC)(x)
        x = AutoAugment()(x)
        x = x.to( torch.float32 )
        x /= torch.max(x)
        x = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)

        return x, y

    def __len__(self):
        return len(self.image_ids)

class TestDataset(Dataset):
    def __init__(self, regressor_dataset, image_dir, device):
        #self.data = data
        #self.label = label
        
        self.image_ids = regressor_dataset[0]
        self.targets = regressor_dataset[1]
        self.images_dir = image_dir
        self.device = device
        
        #######
        # debug
        #self.image_ids = self.image_ids[:100]
        #######
        

    def __getitem__(self, index):
            
        image_id = self.image_ids[index]
        image_path = os.path.join( self.images_dir, image_id )
        image = read_image(image_path, mode=ImageReadMode.RGB)
        y_true = self.targets[index]
        
        y = y_true
        #y = np.log1p( y )
        y = torch.Tensor(y).to(torch.float32)
        y = y.to(self.device)
        
        x = image.to( self.device )
        x = Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC)(x)
        #x = AutoAugment()(x)
        x = x.to( torch.float32 )
        x /= torch.max(x)
        x = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)

        return x, y

    def __len__(self):
        return len(self.image_ids)

class NNRegressor():
    def __init__(self, device="cuda"):
        
        self.device = device
        
        weights = ResNet18_Weights.IMAGENET1K_V1
        #weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights = weights)

        self.composite_model = torch.nn.Sequential(self.model,
                                                   torch.nn.Linear(1000 , 256),
                                                   #torch.nn.Dropout(p=0.5),
                                                   #torch.nn.ReLU(),
                                                   torch.nn.Linear(256, 2),
                                                   #torch.nn.ReLU()
                                                   )

        self.composite_model.to(self.device)
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        pass
    
    def build_dataset(self, train_df):
        
        unique_image_ids = np.array(list(set(train_df["image_id_worm"].to_numpy())))
        n_images = len(unique_image_ids)
        
        image_ids = []
        class_counts = []
        for i in tqdm(range(n_images), desc="Making dataset"):
            
            image_id = unique_image_ids[i]
            image_ids.append( image_id )
            y_true = {"abw": 0, "pbw": 0}
            labeled_subsample = train_df[train_df["image_id_worm"] == image_id]
            for j in range(len(labeled_subsample)):
                labeled_row = labeled_subsample.iloc[j]
                worm_type = labeled_row["worm_type"]
                if worm_type != str("nan"):
                    y_true[worm_type] = labeled_row["number_of_worms"]
            y_true = np.array([ y_true["abw"], y_true["pbw"] ])
            class_counts.append( y_true )    
        class_counts = np.array( class_counts )
        image_ids = np.array( image_ids )
        
        dataset = [ image_ids, class_counts ]
        
        return dataset
    
    def fit(self, regressor_dataset, image_dir, batch_size=16, epochs = 200, learning_rate = 0.001):

        def train_epoch(train_data_loader, loss_function, optimizer):
            size = len(train_data_loader.dataset)
            self.composite_model.train()

            i = 0
            epoch_start = datetime.now()
            for x, y in train_data_loader:
                
                pred = self.composite_model(x)
                loss = loss_function(pred, y)
                
                if str(loss.item()) == "nan":
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                i += 1
                if i % 20 == 0:
                    loss, current = loss.item(), i * len(x)
                    print("loss: {:.5}  [{}/{}] {}".format( loss, current, size, datetime.now() - epoch_start ))
            epoch_end = datetime.now()
            print("Total epoch time: {}".format( epoch_end - epoch_start ))
            
        
        #########
        # debug
        #regressor_dataset[0] = regressor_dataset[0][:320]
        #regressor_dataset[1] = regressor_dataset[1][:320]
        #########
        
        image_ids = list(regressor_dataset[0])
        alias_image_names = []
        alias_targets = []
        detected_images = os.listdir(image_dir)
        for i in tqdm(range(len(image_ids)), desc="Finding alias image names"):
            raw_image_name = image_ids[i].split(".")[0]
            alias_found = False
            for j in range(len(detected_images)):
                if raw_image_name in detected_images[j]:
                    alias_image_name = detected_images[j]
                    alias_image_names.append( alias_image_name )
                    image_ids[i] = alias_image_name
                    alias_found = True
            if alias_found:
                alias_targets.append( regressor_dataset[1][i] )
                
        regressor_dataset[0] = np.array(alias_image_names)
        regressor_dataset[1] = np.array(alias_targets)
        
        self.scaler.fit( regressor_dataset[1] )
        regressor_dataset[1] = self.scaler.transform( regressor_dataset[1] )
        
        train_dataset = TrainDataset(regressor_dataset, image_dir, device="cuda")
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        #loss_function = torch.nn.SmoothL1Loss()
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(self.composite_model.parameters(), lr=learning_rate, weight_decay=1e-2, betas=(0.9, 0.999))
        #optimizer = torch.optim.Adam(self.composite_model.parameters(), lr=learning_rate)
        #optimizer = torch.optim.SGD(self.composite_model.parameters(), lr=learning_rate)

        best_score = np.inf
        best_base_model = None
        best_composite_model = None
        for i in range(epochs):
            print("Epoch: {}".format(i))
            train_epoch(train_data_loader, loss_function, optimizer)
            current_score = self.evaluate(regressor_dataset, image_dir, batch_size=batch_size)
            print("Validation MAE: {}".format(current_score))
            if current_score < best_score:
                print("Previous best MAE: {}".format(best_score))
                best_score = current_score
                best_composite_model = self.composite_model
        self.composite_model = best_composite_model

        pass

    def evaluate(self, regressor_dataset, image_dir, batch_size):

        val_dataset = TestDataset(regressor_dataset, image_dir, device="cuda")
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        y_true = []
        with torch.no_grad():
            for x, y in val_data_loader:
                y = y.cpu().detach().numpy()
                y_true.append(y)
        y_true = np.vstack(y_true)
        y_true = self.scaler.inverse_transform( y_true )

        y_pred = self.predict( regressor_dataset, image_dir, batch_size=batch_size)
        
        fixed_y_true = []
        fixed_y_pred = []
        for i in range(len(y_pred)):
            pred_sum = np.sum(y_pred[i])
            if str(pred_sum) != "nan":
                fixed_y_true.append( y_true[i] )
                fixed_y_pred.append( y_pred[i] )
            
        print("Raw pred len: {} | Min val {} | Max val {}".format(len(y_pred), np.min(y_pred), np.max(y_pred)))
        #print("Fixed pred len: {} | Min val {} | Max val {}".format(len(fixed_y_pred), np.min(fixed_y_pred), np.max(fixed_y_pred)))
        mae_score = mean_absolute_error(fixed_y_true, fixed_y_pred)

        return mae_score

    def predict(self, regressor_dataset, image_dir, batch_size):

        test_dataset = TestDataset(regressor_dataset, image_dir, device="cuda")
        test_dataloader = DataLoader( test_dataset, batch_size=batch_size, shuffle=False )

        self.composite_model.eval()

        y_pred = []
        with torch.no_grad():
            for x, y in test_dataloader:
                x = x.to(self.device)
                pred = self.composite_model(x)
                #pred = torch.exp(pred) - 1.0
                pred = pred.to("cpu").detach().numpy()
                
                y_pred.append(pred)

        y_pred = np.vstack(y_pred)
        
        y_pred = self.scaler.inverse_transform( y_pred )

        return y_pred

    """def get_embeddings(self, x, batch_size):

        test_dataset = TestDataset( x )
        test_dataloader = DataLoader( test_dataset, batch_size=batch_size, shuffle=False )

        self.composite_model.eval()

        y_pred = []
        with torch.no_grad():
            for x in tqdm(test_dataloader, desc="making embeddings"):
                x = x.to(self.device)
                pred = self.base_model(x)
                pred = torch.flatten( pred, 1 )

                pred = pred.to("cpu").detach().numpy()
                y_pred.append( pred )

        y_pred = np.vstack(y_pred)

        return y_pred"""
    
    
    def build_submission_on_raw_test(self, test_df, raw_image_dir, device):
        
        def get_image(image_path):
            
            x = read_image(image_path, mode=ImageReadMode.RGB)
            x = x.to( self.device )
            x = Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC)(x)
            x = x.to( torch.float32 )
            x /= torch.max(x)
            x = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
            x = x[None]
            
            return x
        
        image_ids = test_df["image_id_worm"].to_numpy()
        
        n_images = len(image_ids)
        predicts = []
        for i in tqdm(range(n_images), desc="Building submission"):
            
            image_path = os.path.join( raw_image_dir, image_ids[i] )
            readed_image = get_image( image_path )
            
            y_pred = self.composite_model(readed_image)
            #y_pred = torch.exp(y_pred) - 1.0
            y_pred = y_pred.to("cpu").detach().numpy()
            y_pred = self.scaler.inverse_transform( y_pred )
            y_pred = {"abw": int(y_pred[0][0]), "pbw": int(y_pred[0][1])}

            splitted_image_id = image_ids[i].split(".")
            predict_abw_row = [splitted_image_id[0] + "_abw", y_pred["abw"]]
            predict_pbw_row = [splitted_image_id[0] + "_pbw", y_pred["pbw"]]
            predicts.append( predict_abw_row )
            predicts.append( predict_pbw_row )
        
        submission_df = pd.DataFrame(data=predicts, columns=["image_id_worm", "number_of_worms"])
            
        return submission_df
    
    
    