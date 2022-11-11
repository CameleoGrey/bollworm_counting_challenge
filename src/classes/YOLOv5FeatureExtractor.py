

import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import PILToTensor

from copy import deepcopy
from PIL import Image
from classes.YOLOv5TorchBackend import YOLOv5TorchBackend

class YOLOv5FeatureExtractor():
    def __init__(self, yolov5_model_path, input_size, cutoff=None, device="cuda"):
        
        self.input_size = input_size
        self.cutoff = cutoff
        self.device = device
        self.model = self.build_feature_extractor_model_(yolov5_model_path, cutoff)
        self.model = self.model.to(device)
        
        pass
    
    def build_feature_extractor_model_( self, yolov5_model_path, cutoff ):
        
        yolo_detection_model = YOLOv5TorchBackend(weights=yolov5_model_path, device=torch.device("cpu"))
        
        model = yolo_detection_model.model  # unwrap DetectMultiBackend
        if cutoff is not None:
            model.model = model.model[:cutoff]  # backbone
        
        model.model[-1] = FeatureMapsAverager()  # replace

        #self.stride = model.stride     
        return model
    
    def extract_features(self, image):
        
        #image = deepcopy( image )
        
        if isinstance(image, Image.Image):
            image = self.preprocess_image(image)
        
        if len(image.shape) == 3:
            image = image[None]  # expand for batch dim
        image = image.to(self.device)
        
        self.model.eval()
        
        features = self.model(image).squeeze(0)
        
        features = features.cpu().detach().numpy()
        
        return features
    
    def preprocess_image(self, image):
        image = image.resize(self.input_size, resample = Image.Resampling.BICUBIC )
        image = PILToTensor()(image)
        image = image.float()
        image /= 255.0
        return image

class FeatureMapsAverager(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.f = -1
        self.i = True
        
        pass
    
    def forward(self, x):
        
        result = self.pool(x)
        result = result.flatten(1)
        
        return result