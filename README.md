# Object detection 
Object detection is a computer vision technique that involves identifying and locating objects within an image or video. It goes beyond simply recognizing objects; it also determines their precise location and boundaries.   

# Key Aspects of Object Detection:

Identification: Recognizing the type of object present in the image (e.g., car, person, dog).   

Localization: Pinpointing the exact location of the object within the image using a bounding box.   
# Applications of Object Detection:
Object detection has a wide range of applications, including:

Self-driving cars: Detecting pedestrians, vehicles, and traffic signs.   

Surveillance systems: Monitoring for suspicious activity and identifying individuals.   

Medical image analysis: Detecting tumors or anomalies in medical scans.   

Retail analytics: Tracking customer behavior and inventory management.   

Robotics: Enabling robots to interact with the physical world.   

# Introduction to Yolo

YOLOv5 is a state-of-the-art object detection model that has gained significant popularity due to its speed, accuracy, and ease of use.

This guide provides a comprehensive overview of using YOLOv5 for object detection tasks, focusing on the n, x, l, and 8n variants.

# Key Features of YOLOv5

# Real-time inference:
 YOLOv5 is designed for real-time object detection, making it suitable for various applications.
# High accuracy:
It achieves high accuracy on various object detection benchmarks.
# Efficient architecture:
The model is optimized for efficient inference on different hardware platforms.
# Easy to use:
The Ultralytics library provides a user-friendly interface for training, testing, and deploying YOLOv5 models.
Choosing the Right YOLOv5 Model

# The YOLOv5 family offers models with different trade-offs between speed and accuracy:

YOLOv5n: Smallest and fastest model, suitable for low-power devices.
YOLOv5s: Balanced model, providing a good trade-off between speed and accuracy.
YOLOv5m: Medium-sized model, offering higher accuracy than smaller models.
YOLOv5l: Large model, providing high accuracy but requiring more computational resources.
YOLOv5x: Largest and most accurate model, suitable for high-performance applications.
YOLOv8n: Newest and fastest model, offering a good balance of speed and accuracy.
Using YOLOv5 for Object Detection
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
