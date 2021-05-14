import cv2
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from skimage.measure import compare_ssim, compare_psnr
import argparse

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from sklearn.model_selection import train_test_split

import network
import config as cf
import tools

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('name', help='model name to use training and test')
parser.add_argument('epoch', type=int, help='end epoch num')
parser.add_argument('--folder', default=None, help='add string to predict folder')
args = parser.parse_args()

print(args)

# Model Parameters
model_name = args.name
epoch = args.epoch
pred_str = args.folder

data_dir = '../../cnn-depth-root/data/real/clip_shade/'
data_dir = '../../cnn-depth-root/data/real/shade/'
img_file = data_dir + '{:05d}.png'
model_dir = cf.model_dir + model_name + '/'
save_dir = model_dir + 'save/'
log_file = model_dir + cf.log_file
model_file = save_dir + '/model-%04d.hdf5'
model_final = model_dir + cf.model_final
out_dir = cf.result_dir + model_name + '/'

pred_dir = out_dir + 'pred-%d'%epoch
if not pred_str is None:
    pred_dir += '_' + pred_str
pred_dir += '/'


patch_tl = (0, 0)
img_size = 512
img_size = 1024
img_shape = (img_size, img_size)
img_shape = (1200, 1600)
ch_num = 3
valid_thre = 32 / 255

is_tonemap = True
is_tonemap = False

idx_range = list(range(19))

is_load_min_val = True
is_load_min_val = False
is_load_min_train = True
is_load_min_train = False
is_load_final = True
is_load_final = False

is_dropout = True
# is_dropout = False

def rmse(img1, img2):
    return np.sqrt(np.sum(np.square(img1 - img2)))

def mae(img1, img2):
    return (np.sum(np.abs(img1 - img2)))

def measure(func, **kwargs):
    return func(kwargs['x'], kwargs['x'])

def evaluate(x, y):
    vals = []
    vals.append(measure(mae, x=x, y=y))
    vals.append(measure(rmse, x=x, y=y))
    vals.append(compare_ssim(X=x, Y=y, multichannel=True))
    # vals.append(compare_psnr(x, y))
    return vals

def main():
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    df_log = pd.read_csv(log_file)
    end_point = int(df_log.tail(1).index.values) + 1
    load_epoch = epoch
    if is_load_min_val:
        df_loss = df_log['val_loss']
        load_epoch = df_loss.idxmin() + 1
    elif is_load_min_train:
        df_loss = df_log['loss']
        load_epoch = df_loss.idxmin() + 1
    elif is_load_final:
        load_epoch = end_point

    # model = network.build_unet_model(img_shape, ch_num)
    model = network.build_unet_percuptual_model(img_shape, ch_num)
    # model.summary()
    # print(len(model.layers))
    if is_dropout:
        pred_model = Model(model.input, model.layers[72].output) # with dropout
    else:
        pred_model = Model(model.input, model.layers[58].output) # without dropout

    # model.load_weights(model_final)
    model.load_weights(model_file%load_epoch)

    for idx in tqdm(idx_range):
        img_gray = cv2.imread(img_file.format(idx), 0) / 255
        img_gray = img_gray[:img_shape[0], :img_shape[1]]

        img = cv2.imread(img_file.format(idx), 1) / 255
        img = img[:img_shape[0], :img_shape[1], :]
        # img = tools.tonemap_exr(img[:, :, ::-1])

        img = np.dstack([img_gray, img_gray, img_gray])

        mask = img[:, :, 0] > valid_thre
        mask = img_gray > valid_thre
        mask = np.dstack([mask, mask, mask]).astype('uint8')
            
        pred = pred_model.predict(np.array(img[None, ...]), batch_size=1)

        pred_img = pred[0]
        # pred_img = tools.tonemap_exr(pred_img)
        # pred_img = tools.exr2png(pred_img)
        pred_img = tools.tonemap(pred_img)

        gray_tonemap_img = tools.tonemap(img.astype('float32'))

        pred_img *= mask
        gray_tonemap_img *= mask

        cv2.imwrite(pred_dir + '{:05d}.png'.format(idx), pred_img)
        cv2.imwrite(pred_dir + 'gray_tonemap_{:05d}.png'.format(idx), gray_tonemap_img)

if __name__ == "__main__":
    main()