import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from pathlib import Path
import os
import requests

# folder name = class/label
# file name = image name

def read_images_from_folder(folder_path):
    # only for per folder images reading
    # input - folder_path: folder to read images from
    # output - tuple, form: (folder_name, [list of images])
    return (folder_name, [images])

def read_classification_dataset_generator(dataset_path, batch_size = 50):
    # read per folder images generator
    # input - dataset_path: classification dataset root location, 
    # input - batch_size (optional): how many images to read and return at once
    # output - [{ folder_name : "abc", file_name : "xyz", image : "X10100..."}]
    return [{"label" : label, "image_name" : image_name, "images" : img}]

def read_all_images_in_batch(folder_path, batch_size = 50)
    # read all images from a folder in batch, including subdir also
    # input - folder_path: folder to read image from, like classification dataset, detection dataset
    # input - batch_size: default value is 50, can be set.
    # output - [{folder_name: folder_name, file_name : image_name, image : image}]
    return [{"label": label, "image_name":image_name, "image": img}]
