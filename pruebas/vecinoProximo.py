import imageio as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

image = Image.open('source.jpg')

image = np.array(image)

#dimensiones = image.shape

#altura, ancho = dimensiones[0], dimensiones[1]

#n_altura ,n_ancho = 187,250

#n_image = np.zeros([n_altura,n_ancho, 3], dtype= np.uint8)


print(image.shape)

def vecinoProximo(A,w,h):
    altura, ancho = A.shape[0], A.shape[1] #dimensiones imagen

    new_image = [[A[int(altura * y / h)][int(ancho * x / w)]
                     for x in range(w)] for y in range(h)]

    return np.array(new_image)


def conversion(image):

    return np.mean(image,axis=2,dtype=np.uint8)


#res = cv2.resize(image, dsize=(240, 150), interpolation=cv2.INTER_NEAREST)

image = conversion(image)

res = vecinoProximo(image,150,240)

"""
for x in range (altura):
    for y in range (ancho):

        pass
"""

#print(n_image.shape)

#print(dimensiones)

#print(dimensiones[0], dimensiones[1], dimensiones[2])

#print(dimensiones[0] / 100)

print(type(res))

plt.imshow(res, cmap='gray')
plt.show()