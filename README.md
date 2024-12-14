# Make sure you have to installed ultralytics  and below libraries
pip install -U ultralytics

import torch
import torchvision
import cv2
import random
import os
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd




# loading the pre-trained yolov5 model from pytorch
yolov5 = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
test_img_path = "set/path/to/.jpg"


# using model on img
detected_results = yolov5(test_img_path)
# To see the result  of object detection
detected_results.print()


# To view the img 
detected_results.show()


# Creating dataframe of resulted img to see statistics of each imgs
resultyolov5s_df = detected_results.pandas().xyxy[0] 
resultyolov5s_df


# To make wafer/save file of the result img
 detectedyolov5l_img_result.save()
 rest goes same for each model  same steps 
