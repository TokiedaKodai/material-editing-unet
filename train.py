import cv2
import numpy as np
from itertools import product
from tqdm import tqdm
import sys
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import network
import config as cf
import tools

argv = sys.argv
_, model_name, epoch = argv

epoch = int(epoch)
initial_epoch = 0

data_dir = cf.render_dir + '210317/'
img_file = data_dir + 'img-%d-%s.png'
model_dir = cf.model_dir + model_name + '/'
save_dir = model_dir + 'save/'
log_file = cf.log_file
model_final = model_dir + cf.model_final

list_bsdf = cf.list_bsdf[:3]
x_bsdf = list_bsdf[1:]
y_bsdf = list_bsdf[0]

patch_size = 256
# patch_size = 128
patch_shape = (patch_size, patch_size)
patch_tl = (0, 0)
img_size = 512
ch_num = 3
batch_size = 4

valid_rate = 0.1 # rate of valid pixels to add training patch
valid_thre = 8 # threshold for valid pixel

data_size = 400
learning_rate = 0.001 # Default
dropout_rate = 0.1
val_rate = 0.3
verbose = 1

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
            mask = patch[:, :, 0] > valid_thre
            if np.sum(mask) > patch_size**2 * valid_rate:
                new_list.append(patch)
                list_valid.append(1)
            else:
                list_valid.append(0)
        return new_list, list_valid

    x_data = []
    y_data = []

    for idx in tqdm(idx_range):
        y_img = cv2.imread(img_file%(idx, y_bsdf), 1)
        y_patches = clipPatch(y_img)
        _, valids = selectValidPatch(y_patches)
        
        for bsdf in x_bsdf:
            x_img = cv2.imread(img_file%(idx, bsdf), 1)
            x_patches = clipPatch(x_img)

            for i, is_valid in enumerate(valids):
                if is_valid:
                    x_data.append(x_patches[i])
                    y_data.append(y_patches[i])

    return np.array(x_data), np.array(y_data)

def main():
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
    csv_logger_cb = CSVLogger(model_dir + log_file)

    x_train, y_train = loadImg(range(data_size))
    print('Training data size: ', len(x_train))
    # print(x_train.shape)
    # print(y_train.shape)

    model.fit(
            x_train,
            y_train,
            epochs=epoch,
            batch_size=batch_size,
            initial_epoch=initial_epoch,
            shuffle=True,
            validation_split=val_rate,
            callbacks=[model_save_cb, csv_logger_cb],
            verbose=verbose)

    model.save_weights(model_final)

if __name__ == "__main__":
    main()