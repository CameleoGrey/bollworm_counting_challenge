
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
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.transforms import AutoAugment, Resize, Compose, ConvertImageDtype, PILToTensor, Normalize, InterpolationMode
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

class TrainDataset(Dataset):
    def __init__(self, regressor_dataset, image_dir, device):
        #self.data = data
        #self.label = label
        
        self.cached_images = regressor_dataset[0]
        self.targets = regressor_dataset[1]
        self.images_dir = image_dir
        self.device = device
        
        #######
        # debug
        #self.image_ids = self.image_ids[:100]
        #######
        

    def __getitem__(self, index):
            
            
        """image_id = self.image_ids[index]
        image_path = os.path.join( self.images_dir, image_id + ".pkl" )
        image = load(image_path, verbose=False)
        x = image[1]
        x *= 255.0
        x = x.to( torch.uint8 )
        x = x.to( self.device )
        x = Resize(size=(480, 480), interpolation=InterpolationMode.BICUBIC)(x)"""
        
        x = self.cached_images[index].clone()
        x = x.to( self.device )
        x = AutoAugment()(x)
        x = x.to( torch.float32 )
        #x /= torch.max(x)
        x /= 255.0
        #x = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.480, 0.225])(x)
        
        y_true = self.targets[index]
        y = y_true
        y = torch.Tensor(y).to(torch.float32)
        y = y.to(self.device)

        return x, y

    def __len__(self):
        return len(self.cached_images)

class TestDataset(Dataset):
    def __init__(self, regressor_dataset, image_dir, device):
        #self.data = data
        #self.label = label
        
        self.cached_images = regressor_dataset[0]
        self.targets = regressor_dataset[1]
        self.images_dir = image_dir
        self.device = device
        

    def __getitem__(self, index):
            
        """image_id = self.image_ids[index]
        image_path = os.path.join( self.images_dir, image_id + ".pkl" )
        image = load(image_path, verbose=False)
        x = image[1]
        x *= 255.0
        x = x.to( torch.uint8 )
        x = x.to( self.device )
        x = Resize(size=(480, 480), interpolation=InterpolationMode.BICUBIC)(x)"""
        
        x = self.cached_images[index].clone()
        x = x.to( self.device )
        x = x.to( torch.float32 )
        #x /= torch.max(x)
        x /= 255.0
        #x = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.480, 0.225])(x)
        
        y_true = self.targets[index]
        y = y_true
        y = torch.Tensor(y).to(torch.float32)
        y = y.to(self.device)

        return x, y

    def __len__(self):
        return len(self.cached_images)

class NNHPPredictor():
    def __init__(self, device="cuda"):
        
        self.device = device
        
        #weights = ResNet18_Weights.IMAGENET1K_V1
        #self.model = resnet18(weights = weights)
        
        weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
        self.model = efficientnet_v2_l(weights = weights)

        self.composite_model = torch.nn.Sequential(self.model,
                                                   torch.nn.Linear(1000 , 256),
                                                   #torch.nn.Dropout(p=0.5),
                                                   #torch.nn.ReLU(),
                                                   torch.nn.Linear(256, 6),
                                                   #torch.nn.ReLU()
                                                   )

        self.composite_model.to(self.device)
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        pass
    
    def fit(self, second_order_dataset, image_dir, batch_size=16, epochs = 200, learning_rate = 0.0001):

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
        #second_order_dataset[5] = second_order_dataset[5][:180]
        #second_order_dataset[1] = second_order_dataset[1][:180]
        #########
        
        cached_images = []
        for i in tqdm(range(len(second_order_dataset[5])), desc="Caching images"):
            image_id = second_order_dataset[5][i]
            image_path = os.path.join( image_dir, image_id + ".pkl" )
            image = load(image_path, verbose=False)
            image = image[1]
            image *= 255.0
            image = image.to( torch.uint8 )
            image = Resize(size=(480, 480), interpolation=InterpolationMode.BICUBIC)(image)
            cached_images.append( image )

        
        x_train, x_val, y_train, y_val = train_test_split(cached_images, 
                                                          second_order_dataset[1], 
                                                          shuffle = True, 
                                                          test_size = 0.2,
                                                          random_state = 45)
        
        train_part = [x_train, y_train]
        val_part = [x_val, y_val]
        
        self.scaler.fit( train_part[1] )
        train_part[1] = self.scaler.transform( train_part[1] )
        val_part[1] = self.scaler.transform( val_part[1] )
        
        train_dataset = TrainDataset(train_part, image_dir, device="cuda")
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        #loss_function = torch.nn.SmoothL1Loss()
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(self.composite_model.parameters(), lr=learning_rate, weight_decay=1e-2, betas=(0.9, 0.999))
        #optimizer = torch.optim.Adam(self.composite_model.parameters(), lr=learning_rate)
        #optimizer = torch.optim.SGD(self.composite_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

        best_score = np.inf
        best_base_model = None
        best_composite_model = None
        for i in range(epochs):
            print("Epoch: {}".format(i))
            train_epoch(train_data_loader, loss_function, optimizer)
            current_score = self.evaluate(val_part, image_dir, batch_size=batch_size)
            print("Validation score: {}".format(current_score))
            if current_score < best_score:
                print("Previous best score: {}".format(best_score))
                best_score = current_score
                best_composite_model = deepcopy(self.composite_model).to("cpu")
        self.composite_model = best_composite_model.to(self.device)
        current_score = self.evaluate(val_part, image_dir, batch_size=batch_size)
        print("Final score: {}".format(current_score))

        pass

    def evaluate(self, regressor_dataset, image_dir, batch_size):
        
        self.composite_model.eval()
        
        val_dataset = TestDataset(regressor_dataset, image_dir, device="cuda")
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        y_true = []
        y_pred = []
        with torch.no_grad():
            for x, y in val_data_loader:
                y = y.cpu().detach().numpy()
                y_true.append(y)
                
                x = x.to(self.device)
                pred = self.composite_model(x)
                pred = pred.to("cpu").detach().numpy()
                y_pred.append(pred)
                
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        
        try:
            mse_score = mean_squared_error(y_true, y_pred)
        except Exception as e:
            print(e)
            mse_score = np.inf
        
        return mse_score

    def predict(self, cached_image):

        image = deepcopy( cached_image )
        self.model.eval()
        
        x = image.to( self.device )
        x *= 255.0
        x = x.to( torch.uint8 )
        x = Resize(size=(480, 480), interpolation=InterpolationMode.BICUBIC)(x)
        x = x.to( torch.float32 )
        x /= 255.0
        #x /= torch.max(x)
        #x = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        x = x.unsqueeze(0)
        
        y_pred = self.composite_model(x)
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = self.scaler.inverse_transform( y_pred )
        y_pred = y_pred[0]

        return y_pred
    
    