#!/usr/bin/env python
# coding: utf-8

# ## PyTorch - Pre Trained Models

# In[20]:


import torchvision.models as models
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')
import requests


# In[21]:


#constants
base_path = Path("C:/Users/shubham/Desktop/Masters/Image-Data-Collection/images/image_classification/banana")
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
response = requests.get(LABELS_URL)  
labels = {int(key): value for key, value in response.json().items()}


# In[22]:


def load_imgs():
    img_path = base_path/"banana_01.jpg" 
    return [Image.open(img_path)]

def transform_images(imgs):
    results = []
    for img in imgs:
        results.append(apply_tranformations(img))
    return results

def apply_tranformations(img):
    transform = transforms.Compose([            
        transforms.Resize(256),                    
        transforms.CenterCrop(224),               
        transforms.ToTensor(),                     
        transforms.Normalize(                      
        mean=[0.485, 0.456, 0.406],                
        std=[0.229, 0.224, 0.225]                  
    )])
    transformed_img = transform(img)
    #convert it in to format(batch_size, channel, height, width)
    transformed_img = transformed_img.unsqueeze(0)
    return Variable(transformed_img)

def evaluate_results(model, batch_img):
    model.eval()
    out = model(batch_img)
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    return (labels[index[0].item()], percentage[index[0]].item())

def run_model_inference(model):
    imgs = load_imgs()
    tranformed_imgs = transform_images(imgs)
    #change below line after reading images in  batch, remove [0]
    return evaluate_results(model, tranformed_imgs[0])
    


# #### PTMs
# - resnet18 = models.resnet18(pretrained=True)
# - alexnet = models.alexnet(pretrained=True)
# - squeezenet = models.squeezenet1_0(pretrained=True)
# - vgg16 = models.vgg16(pretrained=True)
# - densenet = models.densenet161(pretrained=True)
# - inception = models.inception_v3(pretrained=True)
# - googlenet = models.googlenet(pretrained=True)
# - shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# - mobilenet = models.mobilenet_v2(pretrained=True)
# - resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
# - wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# - mnasnet = models.mnasnet1_0(pretrained=True)

# In[28]:


model_dict = {
    "resnet18" : models.resnet18(pretrained=True),
    "alexnet" : models.alexnet(pretrained=True),
    "squeezenet" : models.squeezenet1_0(pretrained=True),
    "vgg16" : models.vgg16(pretrained=True),
    "densenet" : models.densenet161(pretrained=True),
    "inception" : models.inception_v3(pretrained=True),
    "googlenet" : models.googlenet(pretrained=True),
    "shufflenet" : models.shufflenet_v2_x1_0(pretrained=True),
    "mobilenet" : models.mobilenet_v2(pretrained=True),
    "resnext50_32x4d" : models.resnext50_32x4d(pretrained=True),
    "wide_resnet50_2" : models.wide_resnet50_2(pretrained=True),
    "mnasnet" : models.mnasnet1_0(pretrained=True),
}


# In[29]:


for model in model_dict:
    (label, label_prob) = run_model_inference(model_dict[model])
    print("{0} : image has {1}% probability of being a {2}".format(model, label_prob, label))


# In[ ]:




