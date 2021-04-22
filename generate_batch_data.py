'''
This is script to generate batch data in order to use data generator in training.
'''

import cv2
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import sys
import os
import argparse

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import network
import config as cf
import tools
import train

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('name', help='save folder name')
parser.add_argument('--batch', type=int, default=16, help='batch size')
args = parser.parse_args()

print(args)

save_name = args.name
batch_size = args.batch # Default 16

data_dir = cf.render_dir + '210324/exr/'
# data_dir = cf.render_dir + '210317/'
img_file = data_dir + 'img-%d-%s.exr'
# img_file = data_dir + 'img-%d-%s.png'
model_dir = cf.model_dir + model_name + '/'
save_dir = model_dir + 'save/'
log_file = model_dir + cf.log_file
model_file = save_dir + '/model-%04d.hdf5'
model_final = model_dir + cf.model_final

list_bsdf = cf.list_bsdf[:3]
list_bsdf = cf.list_bsdf[:2]
x_bsdf = list_bsdf
y_bsdf = list_bsdf[0]
print(list_bsdf)

# Data Parameters
patch_size = 512
# patch_size = 256
# patch_size = 128
patch_shape = (patch_size, patch_size)
patch_tl = (0, 0)
img_size = 512
ch_num = 3
valid_rate = 0.05 # rate of valid pixels to add training patch
valid_thre = 32 / 255 # threshold for valid pixel
# valid_thre = 8
# valid_thre = 0

# Training Parameters
data_size = 400

is_tonemap = True
# is_tonemap = False

def main():
    loss_model = network.build_loss_model(patch_shape, ch_num)

    x_data, y_data = loadImg(range(data_size))
    print('Training data size: ', len(x_data))
    print('X: ', x_data.shape)
    print('Y: ', y_data.shape)

    y_loss_model = loss_model.predict(y_data)