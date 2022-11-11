
import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import color_palette
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.jit import isinstance

from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy 

##########
from classes.paths_config import *
from classes.utils import *
##########

class DatasetBuilder_YOLOv5():
    def __init__(self):
        pass
    
    def build_yolov5_dataset(self, raw_images_dir, parent_build_dir, dataset_build_name, 
                             bbox_df, train_df, 
                             val_part=0.05, random_seed=45,
                             target_image_size = (640, 640)):
        
        self.make_dataset_dirs_hierarchy_(parent_build_dir, dataset_build_name)
        
        self.make_data_yaml_file_(parent_build_dir, dataset_build_name)
        
        tv_image_ids_dict = self.tv_split_dfs_(train_df, val_part=val_part, random_seed=random_seed)
        
        # make common bbox_dict
        bbox_dict = self.make_bbox_dict( bbox_df )
        save( bbox_dict, os.path.join(interim_dir, "bbox_dict") )
        bbox_dict = load( os.path.join(interim_dir, "bbox_dict") )
        
        for split_name in tv_image_ids_dict.keys():
            split_image_ids = tv_image_ids_dict[split_name]
            dataset_dir = os.path.join( parent_build_dir, dataset_build_name )
            self.generate_split_samples_(image_ids_split = split_image_ids,
                                         bbox_dict = bbox_dict,
                                         split_name = split_name, 
                                         dataset_dir = dataset_dir, 
                                         raw_images_dir = raw_images_dir, 
                                         target_image_size = target_image_size)
        
        pass
    
    def generate_split_samples_(self, image_ids_split, bbox_dict, split_name, dataset_dir, raw_images_dir, target_image_size):
        
        split_images_dir = os.path.join( dataset_dir, split_name, "images" )
        split_labels_dir = os.path.join( dataset_dir, split_name, "labels" )
        
        for i in tqdm( range( len( image_ids_split ) ), desc="Building {} YOLOv5 dataset part".format(split_name) ):
            image_id = image_ids_split[i]
            bboxes = bbox_dict[image_id]
            
            # no bounding boxes case
            # save scaled image without labels file 
            if bboxes[0] is None:
                readed_image = self.read_image(raw_images_dir, image_id)
                scaled_image = readed_image.resize(target_image_size, resample = Image.BICUBIC )
                try:
                    self.save_image(scaled_image, split_images_dir, image_id, suffix = "", format=".jpg")
                except Exception as e:
                    print(e)
                    print(image_id)
                    print("do nothing")
                continue
                
            
            # make preprocessed image and bboxes
            readed_image = self.read_image(raw_images_dir, image_id)
            scaled_image, scaled_bboxes = self.resize_image_with_bboxes(readed_image, bboxes, new_size=target_image_size)
            yolov5_bboxes = self.convert_bboxes_to_yolov5_format(scaled_image, scaled_bboxes)
            
            #########
            # only for file names for new dataset!
            # problem with identical image and labels file names
            image_id = image_id.replace(".", "_")
            image_id = image_id.strip()
            #########
            self.save_image(scaled_image, split_images_dir, image_id, suffix = "", format=".jpg")
            
            # make labels file
            worm_classes = {"abw": 0, "pbw": 1}
            labels_file_lines = []
            for bbox in yolov5_bboxes:
                label_string = []
                
                worm_class = worm_classes[ bbox[0] ]
                label_string.append( str(worm_class) )
                
                for j in range(len(bbox[1])):
                    label_string.append( str(bbox[1][j]) )
                
                label_string = " ".join( label_string )
                label_string = label_string.strip()
                label_string = label_string + "\n"
                
                labels_file_lines.append(label_string)
            labels_file_lines[-1] = labels_file_lines[-1].replace("\n", "")
            
            labels_file_path = os.path.join( split_labels_dir, image_id + ".txt" )
            with open(labels_file_path, "w") as labels_file:
                labels_file.writelines( labels_file_lines )
        
        pass
    
    def make_preprocessed_image_(self):
        pass
    
    def make_labels_file_(self):
        pass
    
    def make_dataset_dirs_hierarchy_(self, parent_build_dir, dataset_build_name,):
        
        dataset_dir = os.path.join(parent_build_dir, dataset_build_name)
        if not Path( dataset_dir ).exists():
            Path( dataset_dir ).mkdir(parents=True, exist_ok=True)
        
        sub_dirs = ["train", "valid",]
        sub_dir_inserts = ["images", "labels"]
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(dataset_dir, sub_dir)
            if not Path( sub_dir_path ).exists():
                Path( sub_dir_path ).mkdir(parents=True, exist_ok=True)
                
            for sub_dir_insert in sub_dir_inserts:
                sub_dir_insert_path = os.path.join(sub_dir_path, sub_dir_insert)
                if not Path( sub_dir_insert_path ).exists():
                    Path( sub_dir_insert_path ).mkdir(parents=True, exist_ok=True)
                
        pass
    
    def make_data_yaml_file_(self, parent_build_dir, dataset_build_name):
        
        data_yaml_path = os.path.join( parent_build_dir, dataset_build_name, "dataset_config.yaml" )
        with open(data_yaml_path, "w") as data_yaml_file:
            data_yaml_file.write( "path: {}/{}\n".format(parent_build_dir, dataset_build_name) )
            data_yaml_file.write( "train: train/images\n" )
            data_yaml_file.write( "val: valid/images\n" )
            data_yaml_file.write( "\n" )
            data_yaml_file.write( "names:\n" )
            data_yaml_file.write( "  0: 'abw'\n" )
            data_yaml_file.write( "  1: 'pbw'" )
        
        pass
    
    def tv_split_dfs_(self, df, val_part, random_seed):
        
        unique_image_ids = np.array(list(set(df["image_id_worm"].to_numpy())))
        
        sample_ids = [i for i in range(len(unique_image_ids))]
        np.random.seed(random_seed)
        np.random.shuffle( sample_ids )
        
        train_size = int((1.0 - val_part) * len(df))
        val_size = int(val_part * len(df))
        
        train_ids = sample_ids[ : train_size]
        val_ids = sample_ids[train_size : train_size + val_size]
        
        image_ids_split = {}
        image_ids_split["train"] = unique_image_ids[train_ids]
        image_ids_split["valid"] = unique_image_ids[val_ids]
        
        return image_ids_split
        
        
    
    def make_bbox_dict(self, bbox_df):
        
        image_ids = bbox_df["image_id"].to_numpy()
        worm_types = bbox_df["worm_type"].to_numpy()
        polygons = bbox_df["geometry"].to_numpy()
        
        bbox_dict = {}
        for i in tqdm(range(len(polygons)), desc="Making bbox dict"):
            image_id = image_ids[i]
            if image_id not in bbox_dict.keys():
                bbox_dict[image_id] = []
            
            worm_type = worm_types[i]
            bbox = self.parse_bbox_(polygons[i])
            
            if len(bbox) == 0:
                worm_bbox_pair = None
            else:
                worm_bbox_pair = (worm_type, bbox)
            bbox_dict[image_id].append( worm_bbox_pair )
        
        return bbox_dict
    
    def parse_bbox_(self, polygon_string):
        
        if str(polygon_string) == "nan":
            return []
        if len(polygon_string) == 0:
            return []
        
        polygon_string = polygon_string.replace("POLYGON ((", "")
        polygon_string = polygon_string.replace("))", "")
        polygon_string = polygon_string.split(",")
        
        x_coordinates, y_coordinates = [], []
        for xy in polygon_string:
            xy = xy.strip()
            x, y = xy.split(" ")
            x, y = float( x ), float( y )
            x_coordinates.append(x)
            y_coordinates.append(y)
        
        bbox = [(np.min(x_coordinates), np.min(y_coordinates)),
                (np.max(x_coordinates), np.max(y_coordinates))]
            
        
        return bbox
    
    def read_image(self, image_dir, image_id):
        image_path = os.path.join( image_dir, image_id )
        image = Image.open(image_path)
        return image
    
    def save_image(self, image, dir_to_save, image_id, suffix="", format=".jpg"):
        
        image_save_path = image_id.split(".")
        image_save_path = image_save_path[0] + suffix + format
        image_save_path = os.path.join( dir_to_save, image_save_path )
        image.save(image_save_path)
        
        return self
        
    
    def draw_absolute_bboxes(self, image, bboxes):
        
        image = deepcopy( image )
        draw = ImageDraw.Draw(image)
        #font = ImageFont.truetype(font='arial.ttf', size=(img.size[0] + img.size[1]) // 100)
        
        colors_dict = {"abw": [255, 0, 0], "pbw": [0, 255, 0]}
        
        for bbox in bboxes:
            color = colors_dict[bbox[0]]
            x0, y0 = bbox[1][0][0], bbox[1][0][1]
            x1, y1 = bbox[1][1][0], bbox[1][1][1]
            
            border = [x0, y0, x1, y1]
            thickness = (image.size[0] + image.size[1]) // 1000
            for t in np.linspace(0, 1, thickness):
                border[0], border[1] = border[0] + t, border[1] + t
                border[2], border[3] = border[2] - t, border[3] - t
                draw.rectangle(border, outline=tuple(color))
                
            #text = '{}'.format(bbox[0])
            #text_size = draw.textsize(text, font=font)
            #draw.rectangle([x0, y0 - text_size[1], x0 + text_size[0], y0], fill=tuple(color))
            #draw.text((x0, y0 - text_size[1]), text, fill='black', font=font)
        
        return image
    
    
    def resize_image_with_bboxes(self, image, bboxes, new_size):
        
        image = deepcopy( image )
        
        raw_image_size = image.size
        x_scale = new_size[0] / raw_image_size[0]
        y_scale = new_size[1] / raw_image_size[1]
        
        image = image.resize(new_size, resample = Image.BICUBIC )
        
        bboxes = deepcopy( bboxes )
        for i in range( len(bboxes) ):
            bbox = bboxes[i]
            
            x0, y0 = bbox[1][0][0], bbox[1][0][1]
            x1, y1 = bbox[1][1][0], bbox[1][1][1]
            
            x0, y0 = x0 * x_scale, y0 * y_scale
            x1, y1 = x1 * x_scale, y1 * y_scale
            
            new_bbox = ( bbox[0], [(x0, y0), (x1, y1)] )
            bboxes[i] = new_bbox
        
        return image, bboxes
    
    def convert_bboxes_to_yolov5_format(self, image, bboxes):
        
        
        raw_image_size = image.size
        
        bboxes = deepcopy( bboxes )
        for i in range( len(bboxes) ):
            bbox = bboxes[i]
            
            x0, y0 = bbox[1][0][0], bbox[1][0][1]
            x1, y1 = bbox[1][1][0], bbox[1][1][1]
            
            x_abs_center = (x0 + x1) / 2
            y_abs_center = (y0 + y1) / 2
            x_abs_width = np.max([x0, x1]) - np.min([x0, x1])
            y_abs_height = np.max([y0, y1]) - np.min([y0, y1])
            
            x_rel_center = x_abs_center / raw_image_size[0]
            y_rel_center = y_abs_center / raw_image_size[1]
            x_rel_width = x_abs_width / raw_image_size[0]
            y_rel_height = y_abs_height / raw_image_size[1]
            
            
            new_bbox = ( bbox[0], [x_rel_center, y_rel_center, x_rel_width, y_rel_height] )
            bboxes[i] = new_bbox
        
        return bboxes
    
    def draw_yolov5_bboxes(self, image, bboxes):
        
        image = deepcopy( image )
        draw = ImageDraw.Draw(image)
        
        colors_dict = {"abw": [255, 0, 0], "pbw": [0, 255, 0]}
        
        for bbox in bboxes:
            color = colors_dict[bbox[0]]
            x_rel_center, y_rel_center = bbox[1][0], bbox[1][1]
            x_rel_width, y_rel_height = bbox[1][2], bbox[1][3]
            
            x0 = (x_rel_center - x_rel_width / 2) * image.size[0]
            x1 = (x_rel_center + x_rel_width / 2) * image.size[0]
            y0 = (y_rel_center - y_rel_height / 2) * image.size[1]
            y1 = (y_rel_center + y_rel_height / 2) * image.size[1]
            
            border = [x0, y0, x1, y1]
            thickness = (image.size[0] + image.size[1]) // 1000
            for t in np.linspace(0, 1, thickness):
                border[0], border[1] = border[0] + t, border[1] + t
                border[2], border[3] = border[2] - t, border[3] - t
                draw.rectangle(border, outline=tuple(color))
        
        return image
    