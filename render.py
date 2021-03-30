import numpy as np
import cv2
import pathlib
import shutil

import config as cf
import tools
from tools import printexec
import scene

renderer = cf.renderer
scene_file = cf.scene_file
scene_xml = scene.scene_xml

res = 512

inDir = cf.data_dir + 'small-set-norm-3/'
outDir = cf.render_dir + '210324/'
outFile = cf.render_file
imgFile = outDir + 'png/img-%d-%s.png'
exrFile = outDir + 'exr/img-%d-%s.exr'

list_bsdf = cf.list_bsdf


files = list(pathlib.Path(inDir).glob('*.obj'))
for cnt, fileName in enumerate(files):
	cnt += 400

	fileName = fileName.name
	inFile = inDir + fileName

	for bsdf in list_bsdf:
		f_out = open(scene_file, 'w')
		f_out.write(scene_xml%(inFile, bsdf))
		f_out.close()

		exe_param = '-o ' + outFile + ' ' + scene_file
		printexec(renderer, exe_param)

		shutil.copy(outFile, exrFile%(cnt, bsdf))
		
		img = cv2.imread(outFile, -1)
		img = img[:res, :res, :]
		img = tools.tonemap(img)
		cv2.imwrite(imgFile%(cnt, bsdf), img)