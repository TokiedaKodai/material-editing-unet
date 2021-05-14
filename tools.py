import numpy as np
import cv2
import matplotlib.pyplot as plt
from subprocess import call
import pathlib
from tqdm import tqdm
from itertools import product
import random

from keras.utils import Sequence

import config as cf

# From Config
img_file = cf.img_file
x_bsdf = cf.x_bsdf
y_bsdf = cf.y_bsdf
is_tonemap = cf.is_tonemap

def printexec(cmdstring, paramstring):
    print( cmdstring + ' ' + paramstring)
    call( [cmdstring] + paramstring.strip().split(' ')  )

################ Object Tools ################
def load_vertices(file):
    vertices = []

    for line in open(file, 'r'):
        vals = line.split()

        if not len(vals):
            continue

        if vals[0] is 'v':
            v = vals[1:4]
            vertices.append(v)

    return vertices

def norm_obj(inFilePath, outFilePath):
    vertices = load_vertices(inFilePath)
    vertices = np.array(vertices, dtype=float)
    mean = np.mean(vertices)
    x_mean = np.mean(vertices[:, 0])
    y_mean = np.mean(vertices[:, 1])
    z_mean = np.mean(vertices[:, 2])
    sd = np.sqrt(np.var(vertices))
    
    f_out = open(outFilePath, 'w')
    for line in open(inFilePath, "r"):
        vals = line.split()

        if vals[0] == "v":
            v = vals[1:4]
            v = np.array(v, dtype=float)
            v = [v[0] - x_mean, v[1] - y_mean, v[2] - z_mean] / sd
            vStr = "v %s %s %s\n"%(v[0], v[1], v[2])
            f_out.write(vStr)
        else:
            f_out.write(line)
    f_out.close()

################ Image Tools ################
def tonemap(img, gamma=2.2):
    tm = cv2.createTonemap(gamma=gamma)
    img_tm = tm.process(img)
    return np.clip(img_tm*255, 0, 255).astype('uint8')

def tonemap_exr(img, gamma=2.2):
    tm = cv2.createTonemap(gamma=gamma)
    return tm.process(img)

def exr2png(img):
    return np.clip(img*255, 0, 255).astype('uint8')
    
def plot_graph(df, save_dir, save_name):
    epoch = df['epoch'].values
    train = df['loss'].values
    validation = df['val_loss'].values

    mean = np.mean(train)

    plt.figure()
    plt.plot(epoch, train)
    plt.plot(epoch, validation)
    plt.ylim(0, mean * 2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(save_dir + save_name)

################ Data Loader ################
def loadImg(idx_range):
    def clipPatch(img):
        p_top, p_left = cf.patch_tl
        p_h, p_w = cf.patch_shape
        top_coords = range(p_top, cf.img_size, p_h)
        left_coords = range(p_left, cf.img_size, p_w)

        list_patch = []
        for top, left in product(top_coords, left_coords):
            t, l, h, w = top, left, *cf.patch_shape
            patch = img[t:t + h, l:l + w]
            list_patch.append(patch)
        return list_patch

    def selectValidPatch(list_patch):
        new_list = []
        list_valid = []
        for patch in list_patch:
            # mask = patch[:, :, 0] > valid_thre
            mask = patch[:, :] > cf.valid_thre
            # mask = patch[:, :, 3]
            if np.sum(mask) > cf.patch_size**2 * cf.valid_rate:
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
            y_img = tonemap_exr(y_img)
            y_img = np.nan_to_num(y_img)
            # y_img = y_img[:, :, 0].reshape((img_size, img_size, 1))
        # max_val = np.max(y_img)
        # print('{}: {}, {}'.format(idx, max_val, np.min(y_img)))
        # print(y_img)
        # if not max_val == 0:
        #     y_img /= max_val

        # mask = y_img[:, :, 0] > valid_thre
        # y_img = np.dstack([y_img, mask, mask, mask])

        if cf.is_gray:
            y_img = cv2.cvtColor(y_img, cv2.COLOR_BGR2GRAY)

        y_patches = clipPatch(y_img)
        _, valids = selectValidPatch(y_patches)
        
        for bsdf in x_bsdf:
            x_img = cv2.imread(img_file%(idx, bsdf), -1)
            if is_tonemap:
                x_img = x_img[:, :, ::-1]
                x_img = tonemap_exr(x_img)
                x_img = np.nan_to_num(x_img)
                # x_img = x_img[:, :, 0].reshape((img_size, img_size, 1))
            # if not max_val == 0:
            #     x_img /= max_val

            if cf.is_gray:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)

            x_patches = clipPatch(x_img)

            for i, is_valid in enumerate(valids):
                if is_valid:
                    x_data.append(x_patches[i])
                    y_data.append(y_patches[i])

    return np.array(x_data), np.array(y_data)

################ Batch Generator ################
class BatchGenerator(Sequence):
    def __init__(self, dir_name, data_num):
        self.batches_per_epoch = data_num
        self.x_file = dir_name + cf.np_x_file
        self.y_file = dir_name + cf.np_y_file

    def __getitem__(self, idx):
        random_idx = random.randrange(0, self.batches_per_epoch)
        x = np.load(self.x_file%random_idx, allow_pickle=True)
        y = np.load(self.y_file%random_idx, allow_pickle=True)
        list_y = [f[None, ...] for f in list(y)]
        return x, list_y

    def __len__(self):
        return self.batches_per_epoch

    def on_epoch_end(self):
        pass



if __name__ == "__main__":
    
    inDir = '../data/small-set/'
    outDir = '../data/small-set-norm/'

    files = list(pathlib.Path(inDir).glob('*.obj'))
    for fileName in files:
        fileName = fileName.name
        norm_obj(inDir + fileName, outDir + fileName)
