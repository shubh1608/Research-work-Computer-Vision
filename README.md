# RESEARCH TOPIC - Comparative analysis of pre-trained image classification and object detection models on a custom image dataset
Experimenting with different State of the art Image Classification and Object Detection Deep CNNs on a custom multi domain Image dataset.

## Problem Statement
Deep Learning since its inception has revolutionised the multiple industries. Now, the cars can drive themselves(autonomous driving) and machines can diagnose the disesase(computer aided detection) much faster than humans. Thanks to the advancements made in Deep Learning. But, there are some serious limitations on using Deep Learning. They are data and computation resource hungry, effectively making it tough to adopt it quickly for solving the industry problems. 

These shortcomings can be tackled using techniques such as Transfer Learning, Transfer Learning is the art of using the already trained models(not necessarily on the IID dataset) to enhance the performance of new models. This research work aims to systematically study the transfer learning ability of multiple deep learning models with different image pre-processing techniques so that it can highlight the important insights. These insights can be used as a guide while solving the real world problems.

## Dataset
Custom dataset is prepared for expermenting with the Image Classification and Object detection pre-trained models. Real object images from mutliple domains are collected over the web [bing_image_downloader](https://pypi.org/project/bing-image-downloader/) python module. The downloaded images are then labelled using [makesense.ai](https://www.makesense.ai/) web based tool. It's easy to use and can be used for labelling bounding boxes and even free shapes for image segmentation.

Image classification dataset contains images of single objects and can be found [here](https://drive.google.com/drive/folders/13IFbytK4CiyMcnql_ajE8y4mZFV4K5Lx?usp=sharing), object detection dataset images contains multiple objects per images and can be found [here](https://drive.google.com/drive/folders/1F3LDFb321GDQ0HmSt-e16zjrRNLD6IV1?usp=sharing) along with the annotations.

## Pre-Processing techniques
Total 6 different types of image pre-processing techniques along with more subtypes are analysed in this research work, they are listed below:
1. Color space variations (Gray, color and hue)
2. Sharpening
3. Blurring/De-Noising (Bilateral, Gaussian and Median)
4. Thresholding (Average Thresholding)
5. Morphological Transformations (Opening, Closing)
6. Edge Detections (Canny Edge Detection)

Implementations of all the techniques are used from [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py_table_of_contents_imgproc.html).

## Pre-Trained Models

### Image Classification
20+ pre-trained models along with the architectural variations are analysed in this research work. Models weights are collected from the different software frameworks such as [Keras Applications API](https://keras.io/api/applications/), [PyTorch](https://pytorch.org/vision/stable/models.html).

### Object Detection
10+ object detection models including the single stage and multi stage detectors are analysed. The models were collected from the [Tensorflow2 Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/) and also from the [PyTorch](https://pytorch.org/vision/stable/models.html). Some other open source Github repositories are used for analysing the state of the art models such as PP-YOLO and RetinaNet.

## Results

### Image Classification
EfficientNet family of models were able to perform significantly well, Inception V3 was the runner up with a little less accuracy. The models achieved the highest accuracy on the raw dataset, i.e. no other pre processing techniques helped in gaining the accuracy.

### Object Detection
EfficientDet family of models outperforms others, unlike the case in image classification models, techniques such as Bilateral blurring helped in gaining a slight edge in the model's performance as compared to others.

## Conclusion
EfficientNet family of models has surely impacted the recent state of the art performance and can be used as a good starting point to train a domain/problem specific models. Techniques such as blurring can help in gaining the slight edge in performance hence can be considered especially in object detection.
