renderer = '../Mitsuba 0.5.0/mitsuba'
scene_file = 'scene.xml'

data_dir = '../data/'
render_dir = '../render/'
render_file = 'render.exr'

model_dir = '../models/'
model_file = 'model-%04d.hdf5'
model_final = 'model-final.hdf5'
model_best = 'model-best.hdf5'
log_file = 'training.log'

list_bsdf = ['diffuse', 'dielectric', 'plastic', 'cu', 'au', 'carbon']
list_bsdf = ['diffuse', 'plastic', 'cu2o', 'au']