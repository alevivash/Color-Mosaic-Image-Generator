import imageio as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

def vecinoProximo(A,w,h):
    altura, ancho = A.shape[0], A.shape[1] #dimensiones imagen

    new_image = [[A[int(altura * y / h)][int(ancho * x / w)]
                     for x in range(w)] for y in range(h)]

    return np.array(new_image)


def conversion(image):

    return np.mean(image,axis=2,dtype=np.uint8)



def L1(a,b):

    return abs(np.mean(a) - np.mean(b))



azul = conversion (np.array(Image.open('azul.jpg')))


verde = conversion(np.array(Image.open ('verde.jpg')))

nueva = L1(azul,verde)

print(nueva)

plt.imshow(verde)

plt.show()