import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import image_preprocessing_library as lib
import sys
sys.path.append('C:\\Users\\shubham\\Desktop\\Masters\\code\\common')
print(sys.path)
from common import utils

base_path = "../../Image-Data-Collection/images/image_classification/"
save_location = "../../Image-Data-Collection/images/processed_images/image_classification"

operations_dict = {
    1: ["gray", "bilateral_blur", "canny"],
    2: ["gray", "median_blur", "threshold_otsu"],
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
    dirs = os.listdir(os.path.join(save_location, "apple"))
    return list(dirs)

def process_images(operations_dict):
    exclude_operations = operations_performed()
    for operation_id in operations_dict:
        if str(operation_id) not in exclude_operations:
            for images_tuple in utils.read_class_images_generator(base_path):
                processed_images = apply_operations(images_tuple[1], operations_dict[operation_id])
                utils.save_processed_images(images_tuple[0], operation_id,  processed_images)

process_images(operations_dict)