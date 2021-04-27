renderer = '../Mitsuba 0.5.0/mitsuba'
scene_file = 'scene.xml'

shape_dir = '../shape/'
render_dir = '../render/'
render_file = 'render.exr'

list_bsdf = ['diffuse', 'dielectric', 'plastic', 'cu', 'au', 'carbon']
list_bsdf = ['diffuse', 'plastic', 'cu2o', 'au']

model_dir = '../models/'
model_file = 'model-%04d.hdf5'
model_final = 'model-final.hdf5'
model_best = 'model-best.hdf5'
log_file = 'training.log'

result_dir = '../results/'

#### Data Parameters
data_dir = render_dir + '210324/'
img_file = data_dir + 'exr/img-%d-%s.exr'
np_dir = data_dir + 'np/'
np_train_dir = 'train/'
np_val_dir = 'val/'
np_x_file = 'x-%d.npy'
np_y_file = 'y-%d.npy'

np_data = {
    'name': '100_2m',
    'train': 131,
    'val': 57
}

list_bsdf = list_bsdf[:2]
x_bsdf = list_bsdf
y_bsdf = list_bsdf[0]

#### Training Parameters
patch_size = 512
patch_shape = (patch_size, patch_size)
patch_tl = (0, 0)
img_size = 512
ch_num = 3
valid_rate = 0.05 # rate of valid pixels to add training patch
valid_thre = 32 / 255 # threshold for valid pixel
is_tonemap = True
