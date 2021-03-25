import cv2
import numpy as np
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

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import network
import config as cf
import tools

argv = sys.argv
_, model_name = argv

data_dir = cf.render_dir + '210324/exr/'
img_file = data_dir + 'img-%d-%s.exr'
model_dir = cf.model_dir + model_name + '/'
save_dir = model_dir + 'save/'
log_file = cf.log_file
model_final = model_dir + cf.model_final
out_dir = cf.result_dir + model_name + '/'
pred_dir = out_dir + 'pred/'

list_bsdf = cf.list_bsdf[:4]
x_bsdf = list_bsdf
y_bsdf = list_bsdf[0]

patch_size = 256
patch_shape = (patch_size, patch_size)
patch_tl = (0, 0)
img_size = 512
img_shape = (img_size, img_size)
ch_num = 3

idx_range = range(400, 500)
# idx_range = range(100)

def mse(img1, img2):
    return np.sum(np.square(img1 - img2))

def measure(func, **kwargs):
    return func(kwargs['x'], kwargs['x'])

def evaluate(x, y):
    

def main():
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    model = network.build_unet_model(img_shape, ch_num)
    model.load_weights(model_final)

    for idx in tqdm(idx_range):
        x_test = []
        for bsdf in x_bsdf:
            x_img = cv2.imread(img_file%(idx, bsdf), -1)
            max_val = np.max(x_img)
            if not max_val == 0:
                x_img /= max_val
            x_test.append(x_img)

        y_test = x_test[0]
            
        pred = model.predict(np.array(x_test), batch_size=len(x_bsdf))
        pred = pred.astype('int')
        x_test = np.array(x_test, dtype='int')

        fig, axs = plt.subplots(2, 4, figsize=(15, 10))
        for i in range(4):
            ori_img = x_test[i][:, :, ::-1]
            pred_img = pred[i][:, :, ::-1]

            pred_img = tools.tonemap(pred_img)

            axs[0, i].imshow(ori_img)
            axs[1, i].imshow(pred_img)
            axs[0, i].set_title(x_bsdf[i])
            axs[0, i].axis('off')
            axs[1, i].axis('off')

        plt.savefig(pred_dir + 'pred-{:04d}.png'.format(idx), dpi=300)
        plt.close()

if __name__ == "__main__":
    main()