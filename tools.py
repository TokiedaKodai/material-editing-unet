import numpy as np
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
    # print(mean)
    # print(x_mean)
    # print(y_mean)
    # print(z_mean)
    # print(sd)
    # print(vertices.shape)
    
    f_out = open(outFilePath, 'w')
    for line in open(inFilePath, "r"):
        vals = line.split()

        if vals[0] == "v":
            v = vals[1:4]
            v = np.array(v, dtype=float)
            v = [v[0] - x_mean, v[1] - y_mean, v[2] - z_mean] / sd
            # v *= 100
            vStr = "v %s %s %s\n"%(v[0], v[1], v[2])
            f_out.write(vStr)
        else:
            f_out.write(line)
    f_out.close()

if __name__ == "__main__":
    
    inDir = '../data/small-set/'
    outDir = '../data/small-set-norm/'

    files = list(pathlib.Path(inDir).glob('*.obj'))
    # files = ['1a0c94a2e3e67e4a2e4877b52b3fca7.obj']
    for fileName in files:
        fileName = fileName.name
        norm_obj(inDir + fileName, outDir + fileName)
