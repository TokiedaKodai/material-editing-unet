import numpy as np
import pathlib


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

# def loadOBJ(fliePath):
#     numVertices = 0
#     numUVs = 0
#     numNormals = 0
#     numFaces = 0
#     vertices = []
#     uvs = []
#     normals = []
#     vertexColors = []
#     faceVertIDs = []
#     uvIDs = []
#     normalIDs = []
#     for line in open(fliePath, "r"):
#         vals = line.split()
#         if len(vals) == 0:
#             continue
#         if vals[0] == "v":
#             v = map(float, vals[1:4])
#             vertices.append(v)
#             if len(vals) == 7:
#                 vc = map(float, vals[4:7])
#                 vertexColors.append(vc)
#             numVertices += 1
#         if vals[0] == "vt":
#             vt = map(float, vals[1:3])
#             uvs.append(vt)
#             numUVs += 1
#         if vals[0] == "vn":
#             vn = map(float, vals[1:4])
#             normals.append(vn)
#             numNormals += 1
#         if vals[0] == "f":
#             fvID = []
#             uvID = []
#             nvID = []
#             for f in vals[1:]:
#                 w = f.split("/")
#                 if numVertices > 0:
#                     fvID.append(int(w[0])-1)
#                 # if numUVs > 0:
#                 #     uvID.append(int(w[1])-1)
#                 if numNormals > 0:
#                     nvID.append(int(w[2])-1)
#             faceVertIDs.append(fvID)
#             uvIDs.append(uvID)
#             normalIDs.append(nvID)
#             numFaces += 1

#     return vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors

# def saveOBJ(filePath, vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors):
#     f_out = open(filePath, 'w')
#     f_out.write("####\n")
#     f_out.write("#\n")
#     f_out.write("# Vertices: %s\n" %(len(vertices)))
#     f_out.write("# Faces: %s\n" %(len( faceVertIDs)))
#     f_out.write("#\n")
#     f_out.write("####\n")
#     for vi, v in enumerate( vertices ):
#         vStr = "v %s %s %s"  %(v[0], v[1], v[2])
#         if len( vertexColors) > 0:
#             color = vertexColors[vi]
#             vStr += " %s %s %s" %(color[0], color[1], color[2])
#         vStr += "\n"
#         f_out.write(vStr)
#     f_out.write("# %s vertices\n\n"  %(len(vertices)))
#     for uv in uvs:
#         uvStr =  "vt %s %s\n"  %(uv[0], uv[1])
#         f_out.write(uvStr)
#     f_out.write("# %s uvs\n\n"  %(len(uvs)))
#     for n in normals:
#         nStr =  "vn %s %s %s\n"  %(n[0], n[1], n[2])
#         f_out.write(nStr)
#     f_out.write("# %s normals\n\n"  %(len(normals)))
#     for fi, fvID in enumerate( faceVertIDs ):
#         fStr = "f"
#         for fvi, fvIDi in enumerate( fvID ):
#             fStr += " %s" %( fvIDi + 1 )
#             if len(uvIDs) > 0:
#                 fStr += "/%s" %( uvIDs[fi][fvi] + 1 )
#             if len(normalIDs) > 0:
#                 fStr += "/%s" %( normalIDs[fi][fvi] + 1 )
#         fStr += "\n"
#         f_out.write(fStr)
#     f_out.write("# %s faces\n\n"  %( len( faceVertIDs)) )
#     f_out.write("# End of File\n")
#     f_out.close()

def norm_obj(inFilePath, outFilePath):
    # vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors = loadOBJ(inFilePath)
    vertices = load_vertices(inFilePath)
    vertices = np.array(vertices, dtype=float)
    mean = np.mean(vertices)
    print(mean)
    # saveOBJ(outFilePath, vertices/mean, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors)

    f_out = open(outFilePath, 'w')
    for line in open(inFilePath, "r"):
        vals = line.split()

        if vals[0] == "v":
            v = vals[1:4]
            v = np.array(v, dtype=float) / mean
            vStr = "v %s %s %s\n"%(v[0], v[1], v[2])
            f_out.write(vStr)
        else:
            f_out.write(line + '\n')
    f_out.close()

if __name__ == "__main__":
    
    inDir = '../data/small-set/'
    outDir = '../data/small-set-norm/'

    files = list(pathlib.Path(inDir).glob('*.obj'))
    for fileName in files:
        fileName = fileName.name
        norm_obj(inDir + fileName, outDir + fileName)
