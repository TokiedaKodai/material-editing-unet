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
parser.add_argument('--pred_str', default=None, help='add string to predict folder')
args = parser.parse_args()

print(args)

# Model Parameters
model_name = args.name
epoch = args.epoch
pred_str = args.pred_str

data_dir = cf.render_dir + '210324/exr/'
# data_dir = cf.render_dir + '210317/'
img_file = data_dir + 'img-%d-%s.exr'
# img_file = data_dir + 'img-%d-%s.png'
model_dir = cf.model_dir + model_name + '/'
save_dir = model_dir + 'save/'
log_file = model_dir + cf.log_file
model_file = save_dir + '/model-%04d.hdf5'
model_final = model_dir + cf.model_final
out_dir = cf.result_dir + model_name + '/'

pred_dir = out_dir + 'pred-%d'%epoch
if not pred_str is None:
    pred_dir += '-' + pred_str
pred_dir += '/'

list_bsdf = cf.list_bsdf[:4]
x_bsdf = list_bsdf
y_bsdf = list_bsdf[0]


patch_tl = (0, 0)
img_size = 512
img_shape = (img_size, img_size)
ch_num = 3
valid_thre = 8 / 255

is_tonemap = True
# is_tonemap = False

# idx_range = range(400, 500)
idx_range = list(range(100))
# idx_range.extend(list(range(400, 500)))
idx_range = list(range(2))

is_load_min_val = True
is_load_min_val = False
is_load_min_train = True
is_load_min_train = False
is_load_final = True
is_load_final = False

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
    pred_model = Model(model.input, model.layers[58].output)

    # model.load_weights(model_final)
    model.load_weights(model_file%load_epoch)

    str_results = 'idx,bsdf,mae,rmse,ssim\n'

    for idx in tqdm(idx_range):
        x_test = []
        for bsdf in x_bsdf:
            x_img = cv2.imread(img_file%(idx, bsdf), -1)
            if is_tonemap:
                x_img = x_img[:, :, ::-1]
                x_img = tools.tonemap_exr(x_img)
                x_img = np.nan_to_num(x_img)
            # max_val = np.max(x_img)
            # if not max_val == 0:
            #     x_img /= max_val
            x_test.append(x_img)

        y_test = x_test[0]
        mask = y_test[:, :, 0] > valid_thre
        mask = np.dstack([mask, mask, mask]).astype('float32')
            
        # pred = model.predict(np.array(x_test), batch_size=len(x_bsdf))
        pred = pred_model.predict(np.array(x_test), batch_size=len(x_bsdf))

        for i, bsdf in enumerate(x_bsdf):
            results = evaluate(pred[i], y_test)
            str_results += '{},{},{},{},{}\n'.format(idx, bsdf, results[0], results[1], results[2])

        # pred = pred.astype('int')
        # x_test = np.array(x_test, dtype='int')

        fig, axs = plt.subplots(2, len(x_bsdf), figsize=(15, 10))
        for i in range(len(x_bsdf)):
            # ori_img = x_test[i][:, :, ::-1]
            # pred_img = pred[i][:, :, ::-1]
            ori_img = x_test[i]
            pred_img = pred[i]
            # pred_img *= mask

            # ori_img = x_test[i]
            # pred_img = pred[i]
            # ori_img = tools.tonemap(ori_img)
            # pred_img = tools.tonemap(pred_img)

            # pred_img = tools.tonemap_exr(pred_img)
            # length = np.sum(mask)
            # if not length == 0:
            #     mean_diffuse = np.sum(ori_img * mask) / length
            #     mean_pred = np.sum(pred_img * mask) / length
            #     pred_img = (pred_img / mean_pred) * mean_diffuse

            ori_img = tools.exr2png(ori_img)
            pred_img = tools.exr2png(pred_img)
            
            # pred_img = tools.tonemap(pred_img)

            # pred_img = pred[i][:, :, ::-1]
            # max_pred = np.max(pred_img)
            # if not max_pred == 0:
            #     pred_img = pred_img / max_pred
            # pred_img *= 255
            # pred_img = pred_img.astype('int')

            # max_ori = np.max(ori_img)
            # if not max_ori == 0:
            #     ori_img = ori_img / max_ori
            # ori_img *= 255
            # ori_img = ori_img.astype('int')

            axs[0, i].imshow(ori_img)
            axs[1, i].imshow(pred_img)
            axs[0, i].set_title(x_bsdf[i])
            axs[0, i].axis('off')
            axs[1, i].axis('off')

        plt.savefig(pred_dir + 'pred-{:04d}.png'.format(idx), dpi=300)
        plt.close()
    
    with open(pred_dir + 'results.txt', mode='w') as f:
        f.write(str_results)

if __name__ == "__main__":
    main()