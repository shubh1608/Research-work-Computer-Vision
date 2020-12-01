import os

ROOT_DIR = os.path.dirname(os.path.abspath("config.py"))

config_dict = {

    "root_dir" : ROOT_DIR,

    "image_classification_dataset" : {
        "dataset_path" : os.path.join(ROOT_DIR, "dataset", "image_classification"),
        "annotation_file" : "",
        "labels_file" : "",
        "processed_images_basepath" : os.path.join(ROOT_DIR, "dataset", "processed_images", "image_classification")
    },

    "object_detection_dataset" : {
        "dataset_path" : os.path.join(ROOT_DIR, "dataset", "object_detection", "images"),
        "annotation_file" : os.path.join(ROOT_DIR, "dataset", "object_detection", "annotations", "annotations.csv"),
        "labels_file" : os.path.join(ROOT_DIR, "dataset", "object_detection", "annotations", "labels.txt"),
        "processed_images_basepath" : os.path.join(ROOT_DIR, "dataset", "object_detection", "processed_images", "object_detection")
    },

    "image_preprocessing":{
        
    },
    
    "image_classification":{

    },

    "object_detection":{

    }
}
