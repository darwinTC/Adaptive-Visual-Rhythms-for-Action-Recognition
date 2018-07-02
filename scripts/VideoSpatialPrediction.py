'''
A sample function for classification using spatial network
Customize as needed:
e.g. num_categories, layer for feature extraction, batch_size
'''

import os
import sys
import numpy as np
import math
import cv2
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

sys.path.insert(0, "../../")
import video_transforms


def VideoSpatialPrediction(
        mode,
        vid_name,
        net,
        num_categories,
        start_frame=0,
        num_frames=0,
        num_samples=25,
        index =1,
        new_size = 299
        ):

    if num_frames == 0:
        imglist = os.listdir(vid_name)
        #imglist = list(filter(lambda x: x[:3]=='img',imglist))
        duration = len(imglist)
        # print(duration)
    else:
        duration = num_frames  

    # selection
    if mode == 'rgb':
        step = int(math.floor((duration-1)/(num_samples-1)))
        clip_mean = [0.485, 0.456, 0.406]
        clip_std = [0.229, 0.224, 0.225]
    else:
        clip_mean = [0.5, 0.5]
        clip_std = [0.226, 0.226]

    normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
    val_transform = video_transforms.Compose([
            video_transforms.ToTensor(),
            normalize,
        ])

    deep = 1 if mode == 'rhythm' else 3

    # inception = 320,360, resnet = 240, 320
    width = 320 if new_size==299 else 240
    height = 360 if new_size==299 else 320
    dims = (width,height,deep,num_samples)
    rgb = np.zeros(shape=dims, dtype=np.float64)
    rgb_flip = np.zeros(shape=dims, dtype=np.float64)
   
    for i in range(num_samples):
        if mode == 'rhythm':
            img_file = os.path.join(vid_name, 'visual_rhythm_{0:05d}.jpg'.format(index))
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)   
            img = cv2.resize(img, dims[1::-1])
            rgb[:,:,0,i] = img
            rgb_flip[:,:,0,i] = img[:,::-1]    
        else:        
            img_file = os.path.join(vid_name, 'img_{0:05d}.jpg'.format(i*step+1))
            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, dims[1::-1])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb[:,:,:,i] = img
            rgb_flip[:,:,:,i] = img[:,::-1,:]


    # crop 299 = inception, 224 = resnet
    size = new_size
    corner = [(height-size)//2, (width-size)//2]
    rgb_1 = rgb[:size, :size, :,:]
    rgb_2 = rgb[:size, -size:, :,:]
    rgb_3 = rgb[corner[1]:corner[1]+size, corner[0]:corner[0]+size, :,:]
    rgb_4 = rgb[-size:, :size, :,:]
    rgb_5 = rgb[-size:, -size:, :,:]
    rgb_f_1 = rgb_flip[:size, :size, :,:]
    rgb_f_2 = rgb_flip[:size, -size:, :,:]
    rgb_f_3 = rgb_flip[corner[1]:corner[1]+size, corner[0]:corner[0]+size, :,:]
    rgb_f_4 = rgb_flip[-size:, :size, :,:]
    rgb_f_5 = rgb_flip[-size:, -size:, :,:]

    rgb = np.concatenate((rgb_1,rgb_2,rgb_3,rgb_4,rgb_5,rgb_f_1,rgb_f_2,rgb_f_3,rgb_f_4,rgb_f_5), axis=3)

    _, _, _, c = rgb.shape
    rgb_list = []
    for c_index in range(c):
        cur_img = rgb[:,:,:,c_index]
        cur_img_tensor = val_transform(cur_img)
        rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))
        
    rgb_np = np.concatenate(rgb_list,axis=0)
    batch_size = 25
    prediction = np.zeros((num_categories,rgb.shape[3]))
    num_batches = int(math.ceil(float(rgb.shape[3])/batch_size))

    for bb in range(num_batches):
        span = range(batch_size*bb, min(rgb.shape[3],batch_size*(bb+1)))
        input_data = rgb_np[span,:,:,:]
        imgDataTensor = torch.from_numpy(input_data).type(torch.FloatTensor).cuda()
        imgDataVar = torch.autograd.Variable(imgDataTensor)
        output = net(imgDataVar)
        result = output.data.cpu().numpy()
        prediction[:, span] = np.transpose(result)

    return prediction
