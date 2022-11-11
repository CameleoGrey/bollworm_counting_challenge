
import os
import os
import sys
import pathlib
from pathlib import Path

project_dir = os.path.abspath(os.getcwd())
project_dir = "\\".join( project_dir.split("\\")[:-2] )
sys.path.append( project_dir )

models_dir = os.path.join(project_dir, "models")
if not Path( models_dir ).exists():
    Path( models_dir ).mkdir(parents=True, exist_ok=True)

data_dir = os.path.join(project_dir, "data")
if not Path( data_dir ).exists():
    Path( data_dir ).mkdir(parents=True, exist_ok=True)
    
interim_dir = os.path.join(data_dir, "interim")
if not Path( interim_dir ).exists():
    Path( interim_dir ).mkdir(parents=True, exist_ok=True)

raw_image_dir = os.path.join(data_dir, "images")
if not Path( raw_image_dir ).exists():
    Path( raw_image_dir ).mkdir(parents=True, exist_ok=True)
    
submissions_dir = os.path.join(data_dir, "submissions")
if not Path( submissions_dir ).exists():
    Path( submissions_dir ).mkdir(parents=True, exist_ok=True)

train_cache_dir = os.path.join(data_dir, "train_cache_dir")
if not Path( train_cache_dir ).exists():
    Path( train_cache_dir ).mkdir(parents=True, exist_ok=True)

test_cache_dir = os.path.join(data_dir, "test_cache_dir")
if not Path( test_cache_dir ).exists():
    Path( test_cache_dir ).mkdir(parents=True, exist_ok=True)

plots_dir = os.path.join(data_dir, "plots_dir")
if not Path( plots_dir ).exists():
    Path( plots_dir ).mkdir(parents=True, exist_ok=True)