import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np

# constants, hardcoded values
operations = [
    #color spaces
    "gray",
    "hsv",
    
    #sharpening
    "sharpen",
    
    #thresholding
    "threshold_mean",
    "threshold_gaussian",
    "threshold_otsu",
    
    #smoothing/blurring
    "median_blur",
    "gaussian_blur",
    "bilateral_blur",
    "fastnl_blur",
    
    #morphpological operations
    "erosion",
    "dilation",
    "opening",
    "closing",
    "gradient",
    
    #edge detection
    "sobel",
    "laplacian",
    "canny"
]


# ### Color Spaces
def gray_img(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def hsv_img(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# ### Sharpening
def sharpen_img(img):
    #sharpening using 3x3 array
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1,9,-1], 
                                  [-1,-1,-1]])

    # applying different kernels to the input image
    return cv2.filter2D(img, -1, kernel_sharpening)


# ### Thresholding
# - works only on **gray images**
# - it's good practice to blur images before thresholding, but wont do it in this function as we will test different blurring techniques with different thresholding
# - removing some basic thresholding techniques like binary
def threshold_median(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

def threshold_gaussian(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

def threshold_otsu(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# ### Smoothing/Blurring/Denoising
def median_blur(img):
    return cv2.medianBlur(img,5)

def gaussian_blur(img):
    return cv2.GaussianBlur(img,(5,5),0)

def bilateral_blur(img):
    return cv2.bilateralFilter(img,5,75,75)

def fastnl_blur(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 7, 21, 10, 10)


# ### Morphological Transformations
# - normally performed on binary images
# - Erosion - The basic idea of erosion is just like soil erosion only, it erodes away the boundaries of foreground object (Always try to keep foreground in white), It is useful for removing small white noises
# - Dilation - opposite of erosion, normally dilation is performed after erosion when removing noise from an images.
# - Opening - it does just above mentioned thing, i.e perform dilation after erosion.
# - Closing - Closing is reverse of Opening, Dilation followed by Erosion. It is useful in closing small holes inside the foreground objects, or small black points on the object.
# - Morphological Gradient - difference between dilation and erosion of an image
def erode_img(img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(img, kernel,iterations = 1)

def dilate_img(img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(img,kernel,iterations = 1)

def open_img(img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def close_img(img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def gradient_img(img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)


# ### Edge Detection
# - gaussian smoothing is already done in sobel operation, so first gaussian smoothing then take derivatives to find gradients
def sobel_edge(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobel_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)
    return sobel_OR

def laplacian_edge(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(img,cv2.CV_64F)
    
def canny_edge(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(img,100,200)

#preprocessing function dictionary
dispatcher = {
    #color spaces
    "gray" : gray_img,
    "hsv" : hsv_img,
    
    #sharpening
    "sharpen" : sharpen_img,
    
    #thresholding
    "threshold_mean" : threshold_median,
    "threshold_gaussian" : threshold_gaussian,
    "threshold_otsu" : threshold_otsu,
    
    #smoothing/blurring
    "median_blur" : median_blur,
    "gaussian_blur" : gaussian_blur,
    "bilateral_blur" : bilateral_blur,
    "fastnl_blur" : fastnl_blur,
    
    #morphpological operations
    "erosion" : erode_img,
    "dilation" : dilate_img,
    "opening" : open_img,
    "closing" : close_img,
    "gradient" : gradient_img,
    
    #edge detection
    "sobel" : sobel_edge,
    "laplacian" : laplacian_edge,
    "canny" : canny_edge 
}

