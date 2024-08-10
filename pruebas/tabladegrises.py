import imageio as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img1=np.array([[85,0,127,170,85,150], [119,102,102,123,81,170], [255,170,90,112,63,97], [171,212,225,186,162,171,]], dtype=np.uint8)

print(type(img1))
print(img1.dtype)

print(img1)
print(img1.shape)
print(img1.size)

plt.imshow(img1)
plt.show()