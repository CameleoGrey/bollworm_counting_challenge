
import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

class YOLOv5TorchBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), data=None, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        
        self.pt = True
        
        from yolov5.models.experimental import (attempt_download, attempt_load)

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        w = attempt_download(w)  # download if not local
        fp16 &= True
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA

        model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
        stride = max(int(model.stride.max()), 32)  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        model.half() if fp16 else model.float()
        self.model = model

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16

        y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x
