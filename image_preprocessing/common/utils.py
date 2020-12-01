import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np

classification_base_path = "C:\\Users\\shubham\\Desktop\\Masters\\Pre-Trained-Models-Research-Experiments\\dataset\\image_classification"
classification_save_location = "C:\\Users\\shubham\\Desktop\\Masters\\Pre-Trained-Models-Research-Experiments\\dataset\\processed_images\\image_classification"

# use None when you dont want to limit number of images
per_class_image_read_number = 1

def read_classification_images_generator():
    for dir in os.listdir(classification_base_path):
        if dir == "mixed":
            #to-do - read mixed images
            continue
        images = load_images_from_folder(os.path.join(classification_base_path, dir))
        yield (dir, images)

def read_object_detection_images_generator():
    # to-do : implement here for reading detection images in batch.
    # note : it should return images in tuple form (filename, image), like in class. method
    return None

def load_images_from_folder(folder):
    images = []
    filenames = os.listdir(folder)

    if per_class_image_read_number != None:
        filenames = filenames[:per_class_image_read_number]

    for filename in filenames:
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append((filename, img))
    return images

def save_processed_images(operation_id, label, processed_images):

    location = create_dir(str(operation_id), label)

    for (filename, image) in processed_images:
        cv2.imwrite(os.path.join(location, filename), image)

def create_dir(operation_id, dir):
    folder_path = os.path.join(classification_save_location, operation_id)
    location = None
    if not os.path.exists(folder_path):
        location = os.path.join(classification_save_location, operation_id, dir)
        os.makedirs(location)
    else:
        location = os.path.join(folder_path, dir)
        os.mkdir(location)
    return location