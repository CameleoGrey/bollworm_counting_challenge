
import joblib
import pickle
import numpy as np
from copy import deepcopy
import os
import random


# universal function to serialize object
def save(obj, path, verbose=True):
    if verbose:
        print("Saving object to {}".format(path))

    with open(path, "wb") as obj_file:
        #joblib.dump(obj, obj_file)
        pickle.dump( obj, obj_file, protocol=pickle.HIGHEST_PROTOCOL )

    if verbose:
        print("Object saved to {}".format(path))
    pass

# universal function to deserialize on object
def load(path, verbose=True):
    if verbose:
        print("Loading object from {}".format(path))
    with open(path, "rb") as obj_file:
        #obj = joblib.load(obj_file)
        obj = pickle.load(obj_file)
    if verbose:
        print("Object loaded from {}".format(path))
    return obj