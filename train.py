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
import keras.backend as K
from sklearn.model_selection import train_test_split

import network
import config as cf
import tools as tool

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
parser.add_argument('--aug', action='store_true', help='add to use data augmentation')
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

# From Config
img_file = cf.img_file
np_dir = cf.np_dir
np_train_dir = cf.np_train_dir
np_val_dir = cf.np_val_dir
np_x_file = cf.np_x_file
np_y_file = cf.np_y_file
np_data = cf.np_data
x_bsdf = cf.x_bsdf
y_bsdf = cf.y_bsdf
print('BSDF X', x_bsdf)
print('BSDF Y', y_bsdf)

use_generator = True
# use_generator = False
train_dir = np_dir + np_data['name'] + '/' + np_train_dir
val_dir = np_dir + np_data['name'] + '/' + np_val_dir
train_size = np_data['train']
val_size = np_data['val']

# Model
model_dir = cf.model_dir + model_name + '/'
save_dir = model_dir + 'save/'
log_file = model_dir + cf.log_file
model_final = model_dir + cf.model_final
model_file = save_dir + cf.model_file

# Training Parameters
data_size = 400
# data_size = 100
# data_size = 50
data_size = 2
batch_size = args.batch # Default 16
learning_rate = args.lr # Default 0.001
dropout_rate = args.drop # Default 0.1
val_rate = args.val # Default 0.3
verbose = args.verbose # Default 1

# Augmentation
is_aug = args.aug
augment_rate = 1
shift_max = 0.2

if is_aug:
    datagen_args = dict(
                        width_shift_range=shift_max,
                        height_shift_range=shift_max,
                        shear_range=0,
                        fill_mode='constant',
                        cval=0,
                        )
    x_datagen = ImageDataGenerator(**datagen_args)
    y_datagen = ImageDataGenerator(**datagen_args)
    x_datagen = ImageDataGenerator()
    y_datagen = ImageDataGenerator()
    x_val_datagen = ImageDataGenerator()
    y_val_datagen = ImageDataGenerator()
    seed_train = 1
    seed_val = 2

check_valid = []


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

    if use_generator:
        gen_train = tool.BatchGenerator(train_dir, train_size)
        gen_val = tool.BatchGenerator(val_dir, val_size)
    else:
        # Load Data
        x_data, y_data = tool.loadImg(range(data_size))
        print('Training data size: ', len(x_data))
        print('X: ', x_data.shape)
        print('Y: ', y_data.shape)

        # Generate GT for Perceptual Loss
        loss_model = network.build_loss_model(cf.patch_shape, cf.ch_num)
        y_feature = loss_model.predict(y_data, batch_size=batch_size)

        # Clear Memory
        K.clear_session()

    # model = network.build_unet_model(
    #     patch_shape,
    #     ch_num,
    #     drop_rate=dropout_rate,
    #     lr=learning_rate
    #     )
    model = network.build_unet_percuptual_model(
        cf.patch_shape,
        cf.ch_num,
        drop_rate=dropout_rate,
        lr=learning_rate
        )
    

    model_save_cb = ModelCheckpoint(save_dir + 'model-{epoch:04d}.hdf5',
                                    period=1,
                                    save_weights_only=True)
    csv_logger_cb = CSVLogger(log_file, append=is_model_exist)

    if is_model_exist:
        model.load_weights(model_file%load_epoch)

    

    if is_aug:
        print('data augmentation')
        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, 
                                                        test_size=val_rate, shuffle=False)
        x_datagen.fit(x_train, augment=True, seed=seed_train)
        y_datagen.fit(y_train, augment=True, seed=seed_train)
        x_val_datagen.fit(x_val, augment=True, seed=seed_val)
        y_val_datagen.fit(y_val, augment=True, seed=seed_val)

        x_generator = x_datagen.flow(x_train, batch_size=batch_size, seed=seed_train)
        y_generator = y_datagen.flow(y_train, batch_size=batch_size, seed=seed_train)
        x_val_generator = x_val_datagen.flow(x_val, batch_size=batch_size, seed=seed_val)
        y_val_generator = y_val_datagen.flow(y_val, batch_size=batch_size, seed=seed_val)

        train_generator = zip(x_generator, y_generator)
        val_generator = zip(x_val_generator, y_val_generator)

        model.fit_generator(
                train_generator,
                steps_per_epoch=len(x_train)*augment_rate / batch_size + 1,
                epochs=epoch,
                initial_epoch=init_epoch,
                shuffle=True,
                callbacks=[model_save_cb, csv_logger_cb],
                validation_data=val_generator,
                validation_steps=len(x_val)*augment_rate / batch_size + 1,
                verbose=verbose)
    elif use_generator:
        model.fit_generator(
                gen_train,
                steps_per_epoch=train_size,
                epochs=epoch,
                initial_epoch=init_epoch,
                shuffle=True,
                callbacks=[model_save_cb, csv_logger_cb],
                validation_data=gen_val,
                validation_steps=val_size,
                verbose=verbose)
    else:
        # x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, 
        #                                                 test_size=val_rate, shuffle=False)
        # model.fit(
        #         x_train,
        #         y_train,
        #         epochs=epoch,
        #         batch_size=batch_size,
        #         initial_epoch=init_epoch,
        #         shuffle=True,
        #         validation_data=(x_val, y_val),
        #         callbacks=[model_save_cb, csv_logger_cb],
        #         verbose=verbose)

        # Perceptual Loss
        model.fit(
                x_data,
                y_feature,
                epochs=epoch,
                batch_size=batch_size,
                initial_epoch=init_epoch,
                shuffle=True,
                validation_split=val_rate,
                callbacks=[model_save_cb, csv_logger_cb],
                verbose=verbose)

    model.save_weights(model_final)

    # plot loss graph
    df = pd.read_csv(log_file)
    loss_dir = model_dir + 'loss/'
    os.makedirs(loss_dir, exist_ok=True)
    tool.plot_graph(df, save_dir=loss_dir, save_name='loss_{}.png'.format(epoch))

if __name__ == "__main__":
    main()