import math
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def vecinoProximo(A,w,h):
    altura, ancho = A.shape[0], A.shape[1] #dimensiones imagen

    new_image = [[A[int(altura * y / h)][int(ancho * x / w)]
                     for x in range(w)] for y in range(h)]
    return np.array(new_image)

def initMosaico(source, w, h, p):


    mAltura, mAncho = h * p , w * p


    return vecinoProximo(source,mAncho, mAltura)


def conversion(image):

    return np.mean(image,axis=2,dtype=np.uint8)


image = Image.open('source.jpg')

image = conversion (np.array(image, dtype=np.uint8))

mosaico = initMosaico(image,375,200, 10)


print(mosaico.shape)


plt.imshow(mosaico)
plt.show()

