from PIL import Image
import numpy as np
import imageio as img



image = img.imread('source.jpg')

print(type(image))

print(image.shape)