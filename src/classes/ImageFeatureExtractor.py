
import torch
import torchvision

from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.models import vit_h_14, ViT_H_14_Weights
from copy import deepcopy

class ImageFeatureExtractor():
    def __init__(self, device="cuda"):
        
        #self.weights = ResNet34_Weights.IMAGENET1K_V1
        #self.model = resnet34(weights=self.weights)
        
        weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
        self.model = efficientnet_v2_l(weights = weights)
        self.transforms = weights.transforms()
        self.device = device
        self.model = self.model.to( self.device )
        
        """weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
        self.model = vit_h_14(weights = weights)
        self.transforms = weights.transforms()
        self.device = "cpu"
        self.model = self.model.to( self.device )"""
        
        
        pass
    
    def extract_features(self, image):
        
        image = deepcopy( image )
        
        self.model.eval()
        
        preprocessed_image = self.transforms(image).unsqueeze(0)
        preprocessed_image = preprocessed_image.to( self.device )
        
        features = self.model(preprocessed_image).squeeze(0)
        
        features = features.cpu().detach().numpy()
        
        return features
        