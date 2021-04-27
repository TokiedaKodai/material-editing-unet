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
import tools as tool

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('name', help='save folder name')
parser.add_argument('size', type=int, help='data size')
parser.add_argument('--batch', type=int, default=1, help='batch size')
parser.add_argument('--val', type=float, default=0.3, help='validation data rate')
args = parser.parse_args()

print(args)

save_name = args.name
data_size = args.size
batch_size = args.batch # Default 1
val_rate = args.val # Default 0.3

# From Config
img_file = cf.img_file
np_dir = cf.np_dir
np_train_dir = cf.np_train_dir
np_val_dir = cf.np_val_dir
np_x_file = cf.np_x_file
np_y_file = cf.np_y_file
x_bsdf = cf.x_bsdf
y_bsdf = cf.y_bsdf
print('BSDF X', x_bsdf)
print('BSDF Y', y_bsdf)

save_dir = np_dir + save_name + '/'
save_train = save_dir + np_train_dir
save_val = save_dir + np_val_dir
file_train_x = save_train + np_x_file
file_train_y = save_train + np_y_file
file_val_x = save_val + np_x_file
file_val_y = save_val + np_y_file

os.makedirs(save_train, exist_ok=True)
os.makedirs(save_val, exist_ok=True)

def main():
    loss_model = network.build_loss_model(cf.patch_shape, cf.ch_num)

    x_data, y_data = tool.loadImg(range(data_size))
    print('Data size: ', len(x_data))
    print('X: ', x_data.shape)
    print('Y: ', y_data.shape)

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, 
                                                        test_size=val_rate, shuffle=False)
    train_size = len(x_train)
    val_size = len(x_val)
    print('Train size: ', train_size)
    print('Validation size: ', val_size)

    # Save X
    for i, x in enumerate(x_train):
        np.save(file_train_x%i, x[None, ...])
    for i, x in enumerate(x_val):
        np.save(file_val_x%i, x[None, ...])

    x_train = []
    x_val = []

    # Y
    y_train_f = loss_model.predict(y_train, batch_size=batch_size)#[0]
    # print(y_train_f[0][0,:,:,:].shape)
    for i in range(train_size):
        y = []
        for out in y_train_f:
            print(out[i, :, :, :][None, ...].shape)
            y.append(out[i, :, :, :][None, ...])
        # print(y.shape)
        np.save(file_train_y%i, y)
        # print(np.array(y).shape)

    y_train = []
    y_train_f = []

    # y_val_f = loss_model.predict(y_val, batch_size=batch_size)[0]
    # for i, y in enumerate(y_val_f):
    #     np.save(file_val_y%i, np.array(y[None, ...]))

    y_val_f = loss_model.predict(y_val, batch_size=batch_size)
    for i in range(val_size):
        y = []
        for out in y_val_f:
            y.append(out[i, :, :, :][None, ...])
        np.save(file_val_y%i, np.array(y))


if __name__ == "__main__":
    main()