import cv2
import shutil

import tools
import config as cf

data_dir = cf.render_dir + '210324/'
in_file = data_dir + 'png/img-%d-%s.png'
out_A = data_dir + 'testA/img-%d-%s.png'
out_B = data_dir + 'testB/img-%d-%s.png'
bsdf_A = ['diffuse']
bsdf_B = ['plastic']

# for idx in range(400, 500):
#     for bsdf in bsdf_A:
#         shutil.copy(in_file%(idx, bsdf), out_A%(idx, bsdf))
#     for bsdf in bsdf_B:
#         shutil.copy(in_file%(idx, bsdf), out_B%(idx, bsdf))

