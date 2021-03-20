# material-editing-unet

Learning based material editing using U-net architecture.
Network estimates the material of input image and outputs edited image with target material.
Diffuse and specular are also supported, and only the material can be changed without changing the other properties (shape and illumination).
This is supervised method and needs image pairs of target material and other materials for training.

## Prepare data

You can prepare training and test data by yourself, also can generate images by Mitsuba renderer and ShapeNet (3D model database).
It normalizes the shapes of ShapeNet and automatically renders the shapes with different material you select.

## Training

