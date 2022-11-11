
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from classes.paths_config import *
from classes.utils import *
from classes.DatasetBuilder_YOLOv5 import DatasetBuilder_YOLOv5

import yolov5

dataset_builder = DatasetBuilder_YOLOv5()

bbox_df = pd.read_csv(os.path.join( data_dir, "images_bboxes.csv" ))
bbox_dict = dataset_builder.make_bbox_dict( bbox_df )
save( bbox_dict, os.path.join(interim_dir, "bbox_dict") )

bbox_dict = load( os.path.join(interim_dir, "bbox_dict") )
print(len(bbox_dict.keys()))

image_id = "id_00adaccff7693b67eeb8fdc7.jpg"
bboxes = bbox_dict[image_id]
readed_image = dataset_builder.read_image(raw_image_dir, image_id)

image_with_bboxes = dataset_builder.draw_absolute_bboxes(readed_image, bboxes)
dataset_builder.save_image(image_with_bboxes, interim_dir, image_id, suffix = "_raw_bboxes")

scaled_image, scaled_bboxes = dataset_builder.resize_image_with_bboxes(readed_image, bboxes, new_size=(640, 640))
image_with_bboxes = dataset_builder.draw_absolute_bboxes(scaled_image, scaled_bboxes)
dataset_builder.save_image(image_with_bboxes, interim_dir, image_id, suffix = "_scaled_bboxes")

yolov5_bboxes = dataset_builder.convert_bboxes_to_yolov5_format(readed_image, bboxes)
image_with_bboxes = dataset_builder.draw_yolov5_bboxes(readed_image, yolov5_bboxes)
dataset_builder.save_image(image_with_bboxes, interim_dir, image_id, suffix = "_raw_yolov5_bboxes")

scaled_image, scaled_bboxes = dataset_builder.resize_image_with_bboxes(readed_image, bboxes, new_size=(640, 640))
yolov5_bboxes = dataset_builder.convert_bboxes_to_yolov5_format(scaled_image, scaled_bboxes)
image_with_bboxes = dataset_builder.draw_yolov5_bboxes(scaled_image, yolov5_bboxes)
dataset_builder.save_image(image_with_bboxes, interim_dir, image_id, suffix = "_scaled_yolov5_bboxes")

print("done")