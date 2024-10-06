import numpy as np
import glob, cv2
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import albumentations.pytorch
import torch
import torch.distributed as dist
import torch.nn as nn

import torchvision
import os
import random
import torch.utils.data as data_utils
import datetime
from shutil import copyfile
import time

import shutil 
import scipy
import torch.nn.functional as F
from ultralytics import YOLO

model = YOLO('./runs/nonE2E_Data/weights/best.pt').cuda()
print(model.task)
print(model.model.end2end)

try: 
    shutil.rmtree('./Result/')
    print("Removed New dir")
except:
    print("Making New dir")

filepath = './Result/'
if not os.path.exists(os.path.dirname(filepath)):
    try:
        os.makedirs(os.path.dirname(filepath))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


# Define the base directories
input_base_dir = ''#Val Data directory
output_base_dir = './Result/'

t1 = []
t2 = []
t3 = []
t4 = []
# Iterate over all PNG images in the specified directory structure
if len(glob.glob(os.path.join(input_base_dir, '*', 'img', '*.png')))<100:
    print("Error : Directory Invalid")
    
for img_name in glob.glob(os.path.join(input_base_dir, '*', 'img', '*.png')):
    
    # Extract city name and file name
    city_name = os.path.basename(os.path.dirname(os.path.dirname(img_name)))
    file_name = os.path.splitext(os.path.basename(img_name))[0]
    
    # Define the output text file path
    output_dir = os.path.join(output_base_dir, city_name, 'txt')
    output_file = os.path.join(output_dir, f"{file_name}.txt")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    

    
    #===============================
    t1.append(time.time())
    #===============================
    
    results = model(img_name,verbose=False, imgsz=[480,1280],conf=0.001)

    #===============================
    t2.append(time.time())
    #===============================
    target_outputs = results[0].boxes.data.cpu().numpy()
    target_img = results[0].orig_img

    target_img = np.array(target_img[:, :, ::-1])

    xyxy = target_outputs[:,0:4]
    cls = target_outputs[:,5].astype('int')
    loc = target_outputs[:,6].astype('int')
    action = target_outputs[:,7:].astype('int')
    #===============================
    t3.append(time.time())
    #===============================

    
    with open(output_file, 'w') as f:
        for i in range(xyxy.shape[0]):
            #print(naming_num[i])

            result_box = results[0].boxes.data[i].cpu().numpy()
            result_mask = results[0].masks.xyn[i].flatten()
            if len(result_mask) >5:
         
                result_txt = str(result_box[4])+' '+np.array2string(result_box[5:].astype(int))[1:-1]+' '+' '.join(map(str, result_mask))+'\n'
                f.write(result_txt)

    #===============================

    t4.append(time.time())
    
    
print('Run model              :' , np.median((np.array(t2)-np.array(t1))))
print('Post Process           :' , np.median((np.array(t3)-np.array(t2))))
print('Plot Results           :' , np.median((np.array(t4)-np.array(t3))))