import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import image_preprocessing_library as lib
import sys
sys.path.append('C:\\Users\\shubham\\Desktop\\Masters\\code\\common')
from common import utils

TRACKER_FILE_LOCATION = "C:\\Users\\shubham\\Desktop\\Masters\\Pre-Trained-Models-Research-Experiments\\dataset\\processed_images\\image_classification\\tracker.txt" 

#open tracker file
tracker = open(TRACKER_FILE_LOCATION, "r+")

operations_dict = {
    0 : [], # meant for no operations, i.e have raw images
    1 : ["gray", "bilateral_blur", "canny"],
    2 : ["gray", "median_blur", "threshold_otsu"],
    3 : ["gray"]
}

def apply_operations(images_tuple, operations):
    processed_images = []
    for (filename, image) in images_tuple:
        processed_image = image
        for operation in operations:
            processed_image = lib.dispatcher[operation](processed_image)
        processed_images.append((filename, processed_image))
    return processed_images

def operations_performed():
    exclude_operations = tracker.read().splitlines()
    return exclude_operations

def log_operations(operation_id):
    logged_operations = tracker.read().splitlines()
    if str(operation_id) not in logged_operations:
        tracker.write(str(operation_id) + "\n")


def process_images(operations_dict):
    exclude_operations = operations_performed()
    for operation_id in operations_dict:
        if str(operation_id) not in exclude_operations:
            for images_tuple in utils.read_classification_images_generator():
                processed_images = apply_operations(images_tuple[1], operations_dict[operation_id])
                utils.save_processed_images(operation_id, images_tuple[0],  processed_images)
            log_operations(operation_id)

process_images(operations_dict)