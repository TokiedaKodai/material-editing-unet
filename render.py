import numpy as np
import cv2
import pathlib

import config as cf
import tools
from tools import printexec
import scene

renderer = cf.renderer
scene_file = cf.scene_file
scene_xml = scene.scene_xml

res = 512

inDir = cf.data_dir + 'small-set-norm/'
outDir = cf.render_dir + '210323/'
outFile = cf.render_file
imgFile = outDir + 'img-%d-%s.png'

list_bsdf = cf.list_bsdf


files = list(pathlib.Path(inDir).glob('*.obj'))
for cnt, fileName in enumerate(files):
	cnt += 0

	fileName = fileName.name
	inFile = inDir + fileName

	for bsdf in list_bsdf:
		f_out = open(scene_file, 'w')
		f_out.write(scene_xml%(inFile, bsdf))
		f_out.close()

		exe_param = '-o ' + outFile + ' ' + scene_file
		printexec(renderer, exe_param)
		
		img = cv2.imread(outFile, -1)
		img = img[:res, :res, :]
		img = tools.tonemap(img)
		cv2.imwrite(imgFile%(cnt, bsdf), img)