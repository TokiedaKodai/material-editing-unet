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

''' argv
model name
epoch
is model exist
'''

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('name', help='model name to use training and test')
parser.add_argument('epoch', type=int, help='end epoch num')
parser.add_argument('--exist', action='store_true', help='add, if pre-trained model exist')
parser.add_argument('--min_train', action='store_true', help='add to re-train from min train loss')
parser.add_argument('--min_val', action='store_true', help='add to re-train from min val loss')
parser.add_argument('--batch', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--drop', type=float, default=0.1, help='dropout rate')
parser.add_argument('--val', type=float, default=0.3, help='validation data rate')
parser.add_argument('--verbose', type=int, default=1, help='[0 - 2]: progress bar')
args = parser.parse_args()

print(args)

# Model Parameters
model_name = args.name
epoch = args.epoch
is_model_exist = args.exist
is_load_min_train = args.min_train
is_load_min_val = args.min_val

data_dir = cf.render_dir + '210324/exr/'
img_file = data_dir + 'img-%d-%s.exr'
model_dir = cf.model_dir + model_name + '/'
save_dir = model_dir + 'save/'
log_file = model_dir + cf.log_file
model_file = save_dir + '/model-%04d.hdf5'
model_final = model_dir + cf.model_final

list_bsdf = cf.list_bsdf[:3]
x_bsdf = list_bsdf
y_bsdf = list_bsdf[0]

# Data Parameters
patch_size = 256
# patch_size = 128
patch_shape = (patch_size, patch_size)
patch_tl = (0, 0)
img_size = 512
ch_num = 3
valid_rate = 0.05 # rate of valid pixels to add training patch
valid_thre = 8 / 255 # threshold for valid pixel

# Training Parameters
data_size = 400
batch_size = args.batch # Default 4
learning_rate = args.lr # Default 0.001
dropout_rate = args.drop # Default 0.1
val_rate = args.val # Default 0.3
verbose = args.verbose # Default 1

scaling = 1
is_tonemap = True

def loadImg(idx_range):
    def clipPatch(img):
        p_top, p_left = patch_tl
        p_h, p_w = patch_shape
        top_coords = range(p_top, img_size, p_h)
        left_coords = range(p_left, img_size, p_w)

        list_patch = []
        for top, left in product(top_coords, left_coords):
            t, l, h, w = top, left, *patch_shape
            patch = img[t:t + h, l:l + w]
            list_patch.append(patch)
        return list_patch

    def selectValidPatch(list_patch):
        new_list = []
        list_valid = []
        for patch in list_patch:
            # mask = patch[:, :, 0] > valid_thre
            mask = patch[:, :, 3]
            if np.sum(mask) > patch_size**2 * valid_rate:
                new_list.append(patch)
                list_valid.append(1)
            else:
                list_valid.append(0)
        return new_list, list_valid

    x_data = []
    y_data = []

    for idx in tqdm(idx_range):
    # for idx in idx_range:
        y_img = cv2.imread(img_file%(idx, y_bsdf), -1)
        if is_tonemap:
            y_img = y_img[:, :, ::-1]
            y_img = tools.tonemap_exr(y_img)
            y_img = np.nan_to_num(y_img)
        # max_val = np.max(y_img)
        # print('{}: {}, {}'.format(idx, max_val, np.min(y_img)))
        # print(y_img)
        # if not max_val == 0:
        #     y_img /= max_val

        mask = y_img[:, :, 0] > valid_thre
        y_img = np.dstack([y_img, mask, mask, mask])

        y_patches = clipPatch(y_img)
        _, valids = selectValidPatch(y_patches)
        
        for bsdf in x_bsdf:
            x_img = cv2.imread(img_file%(idx, bsdf), -1)
            if is_tonemap:
                x_img = x_img[:, :, ::-1]
                x_img = tools.tonemap_exr(x_img)
                x_img = np.nan_to_num(x_img)
            # if not max_val == 0:
            #     x_img /= max_val
            x_patches = clipPatch(x_img)

            for i, is_valid in enumerate(valids):
                if is_valid:
                    x_data.append(x_patches[i])
                    y_data.append(y_patches[i])

    return np.array(x_data), np.array(y_data)

def main():
    init_epoch = 0
    if is_model_exist:
        df_log = pd.read_csv(log_file)
        end_point = int(df_log.tail(1).index.values) + 1
        init_epoch = end_point
        load_epoch = end_point
        if is_load_min_val:
            df_loss = df_log['val_loss']
            load_epoch = df_loss.idxmin() + 1
        elif is_load_min_train:
            df_loss = df_log['loss']
            load_epoch = df_loss.idxmin() + 1

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    model = network.build_unet_model(
        patch_shape,
        ch_num,
        drop_rate=dropout_rate,
        lr=learning_rate
        )
    model_save_cb = ModelCheckpoint(save_dir + 'model-{epoch:04d}.hdf5',
                                    period=1,
                                    save_weights_only=True)
    csv_logger_cb = CSVLogger(log_file)

    if is_model_exist:
        model.load_weights(model_file%load_epoch)

    x_train, y_train = loadImg(range(data_size))
    print('Training data size: ', len(x_train))

    x_train *= scaling
    y_train *= scaling

    model.fit(
            x_train,
            y_train,
            epochs=epoch,
            batch_size=batch_size,
            initial_epoch=init_epoch,
            shuffle=True,
            validation_split=val_rate,
            callbacks=[model_save_cb, csv_logger_cb],
            verbose=verbose)

    model.save_weights(model_final)

if __name__ == "__main__":
    main()