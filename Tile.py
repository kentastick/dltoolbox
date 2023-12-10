import numpy as np 
import pandas as pd 
import glob
import openslide
import matplotlib.pyplot as plt
import cv2,math,sys,time,os
import PIL.Image as Image
from tqdm import tqdm
import warnings
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import offsetbox
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
import scipy
from sklearn.model_selection import train_test_split
import timm
import shutil
import h5py
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import pickle

class tile():
    def __init__(self, wsi_path="", save_path="./tile", level=3, window_size=224, stride_size=224):
        self.wsi_path = wsi_path
        self.save_path = save_path
        self.level=level
        self.window_size = window_size
        self.stride_size = stride_size
        self.threshold = 188
        self.df_com = pd.DataFrame()
    def __len__(self):
        print(os.listdir(self.save_path))
        return len(os.listdir(self.save_path))
    
    def __call__(self):

        self.df_com["gray_value"]
    
    def create_tile(self, reset=False):
        wsi_list = glob.glob(f"{self.wsi_path}/*/*ndpi")
        print(wsi_list)
        if reset :shutil.rmtree(self.save_path, ignore_errors=True)
        for path in wsi_list:
            label = path.split("\\")[-2]
            filename=os.path.basename(path).replace(".ndpi", "")
            if os.path.isdir(f"{self.save_path}/{filename}"):continue
            slide = openslide.OpenSlide(path)
            img = np.array(slide.read_region(location= (0,0), level=5, size=slide.level_dimensions[5]))[...,:3]
            img_ratio = slide.level_dimensions[5][0]/slide.level_dimensions[self.level][0]
            #img=cv2.resize(img, None, fx=0.3,fy=0.3)
            mono = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            r, dst = cv2.threshold(mono, 0, 255, cv2.THRESH_OTSU)
            df = generate_patch(path, self.save_path,self.level,self.window_size,self.stride_size, threshold = r)
            df["label"] = label
            df["thresh"] = r
            x_list = df["xtop_left_pixel"]
            y_list = df["ytop_left_pixel"]
            cv2.imwrite(f"{self.save_path}/{filename}_original.jpg", img)
            cv2.imwrite(f"{self.save_path}/{filename}_mask.jpg", dst)
            cv2.imwrite(f"{self.save_path}/{filename}_tile.jpg", self.insert_mask(img,[(int(x*img_ratio),int(y*img_ratio)) for x,y in zip(x_list, y_list)],img_ratio))
            self.df_com = pd.concat([self.df_com,df])
        
    def insert_mask(self,image, coordinate_list,ratio):
        img_bl = np.empty(image.shape)
        window_size = int(self.stride_size*ratio)
        for patch in coordinate_list:
            x = patch[0] 
            y = patch[1]
            img_bl[
                x : x + window_size,
                y : y+  window_size,
                :
            ] = 255
        return(img_bl)

    def save(self,path=""):
        path = path if path else f"{self.save_path}/project.pkl"
        data = {}
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if "__" not in attribute_name and not callable(attribute):
                data[attribute_name] = attribute
            with open(path, 'wb') as f:
                pickle.dump(data, f)
    def load(self,path=""):
        path = path if path else self.save_path
        with open(path, 'rb') as f:
            data = pickle.load(f)
            for key,value in data.items():
                setattr(self, key, value)
        
   
def get_mnist():
    train_data = torchvision.datasets.MNIST(
        root='..',
        train=True,                                     # this is training data
        transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
            # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=True,                        # download it if you don't have it
    )
    return train_data


def compute_statistics(image):
    width, height = image.shape[0], image.shape[1]
    num_pixels = width * height
    
    num_white_pixels = 0
    
    summed_matrix = np.sum(image, axis=-1)
    # Note: A 3-channel white pixel has RGB (255, 255, 255)
    num_white_pixels = np.count_nonzero(summed_matrix > 620)
    ratio_white_pixels = num_white_pixels / num_pixels
    gray = image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722
    gray_value=gray.astype(np.uint8).mean()
    
    green_concentration = np.mean(image[1])
    blue_concentration = np.mean(image[2])
    
    return ratio_white_pixels, green_concentration, blue_concentration, gray_value

def generate_patch(slide_path,export_path ="",level =1 ,window_size=200, stride=128, threshold=188):
    filename = os.path.basename(slide_path).replace(".ndpi", "")
    os.makedirs(f"{export_path}/{filename}", exist_ok=True)
    slide = openslide.OpenSlide(slide_path)
    slide_level= slide.level_dimensions[level]
    image = np.array(slide.read_region((0,0),level,slide_level))
    max_width, max_height = image.shape[0], image.shape[1]
    regions_container = []
    df = pd.DataFrame()
   
    count=0
    i = 0
    while window_size + stride*i <= max_height:
        j = 0
        while window_size + stride*j <= max_width:            
            
            x_top_left_pixel = j * stride
            y_top_left_pixel = i * stride
            patch = image[
                x_top_left_pixel : x_top_left_pixel + window_size,
                y_top_left_pixel : y_top_left_pixel + window_size,
                :3
            ]
            ratio_white_pixels, green_concentration, blue_concentration,gray_value = compute_statistics(patch)
            tile_path = f"{export_path}/{filename}/{filename}_{x_top_left_pixel}_{y_top_left_pixel}_{window_size}.png"
            if gray_value < threshold:    
                region_tuple = (x_top_left_pixel, y_top_left_pixel, ratio_white_pixels, green_concentration, blue_concentration, gray_value, tile_path)
                regions_container.append(region_tuple)
                cv2.imwrite(tile_path, patch)
                count += 1
            j += 1
        i += 1
        
        #outfh.create_dataset(name = "images",data= np.array(image_container),  dtype=np.uint8)
    df["id"] = [filename]*count
    df["tile_path"] =  [a[6] for a in regions_container]  
    df["xtop_left_pixel"] = [a[0] for a in regions_container]
    df["ytop_left_pixel"] = [a[1] for a in regions_container]
    df["ratio_white_pixel"] = [a[2] for a in regions_container]
    df["green_concentration"] = [a[3] for a in regions_container]
    df["blue_concentration"] = [a[4] for a in regions_container]
    df["gray_value"] = [a[5] for a in regions_container]
    df["ndpi_path"] = slide_path
    df["patch_size"] = window_size
    df["stride_size"] = stride
    df["level"] = level
    return df

