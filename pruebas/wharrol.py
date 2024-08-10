import imageio as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image = img.imread('source.jpg')

T = np.shape(image)

print(T)

rojo = np.zeros(T)
verde = np.zeros(T)
azul = np.zeros(T)
morado = np.zeros(T)

for i in range (T[0]):
    for j in range (T[1]):
        rojo[i,j,0] = image[i,j,0]
        verde[i, j, 1] = image[i, j, 1]
        azul[i, j, 2] = image[i, j, 2]
        morado[i, j, 0], morado[i, j, 2] = image[i,j,0], image[i,j,2]

rojo = np.uint8(rojo)
verde = np.uint8(verde)
azul = np.uint8(azul)
morado = np.uint8(morado)

plt.figure()
plt.subplot(2, 2, 1)
plt.axis('off')
plt.imshow(rojo)
plt.subplot(2, 2, 2)
plt.axis('off')
plt.imshow(verde)
plt.subplot(2, 2, 3)
plt.axis('off')
plt.imshow(azul)
plt.subplot(2, 2, 4)
plt.axis('off')
plt.imshow(morado)
plt.show()

