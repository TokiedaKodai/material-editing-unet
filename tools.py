import numpy as np
import cv2
from subprocess import call
import pathlib

def printexec(cmdstring, paramstring):
    print( cmdstring + ' ' + paramstring)
    call( [cmdstring] + paramstring.strip().split(' ')  )

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

def tonemap(img, gamma=2.2):
    tm = cv2.createTonemap(gamma=gamma)
    img_tm = tm.process(img)
    return np.clip(img_tm*255, 0, 255).astype('uint8')

if __name__ == "__main__":
    
    inDir = '../data/small-set/'
    outDir = '../data/small-set-norm/'

    files = list(pathlib.Path(inDir).glob('*.obj'))
    for fileName in files:
        fileName = fileName.name
        norm_obj(inDir + fileName, outDir + fileName)
