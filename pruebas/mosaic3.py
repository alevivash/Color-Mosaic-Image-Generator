import imageio as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

def L1(a,b):

    return abs(np.mean(a) - np.mean(b))

def conversion(image):

    return np.mean(image,axis=2,dtype=np.uint8)

def vecinoProximo(A,w,h):

    A = np.array(A)

    altura, ancho = A.shape[0], A.shape[1] #dimensiones imagen

    new_image = [[A[int(altura * y / h)][int(ancho * x / w)]
                     for x in range(w)] for y in range(h)]

    return np.array(new_image)

def piezas(image, w, h): #--->lista de imagenes en partes, h y w dimensiones piezas

    image = Image.fromarray(image)

    width, height = image.size

    partes = []
    for x in range(0, width, w):
        for y in range(0, height, h):
            # print([x,y,x+w,y+h])
            partes.append((x, y, x + w, y + h))


    for i in range(len(partes)):
        partes[i] = image.crop(partes[i])
        partes[i] = np.array(partes[i])

    return partes  # ---> lista de la imagen en porciones np.array

def initMosaico(source, w, h, p):


    mAltura, mAncho = h * p , w * p


    return vecinoProximo(source,mAncho, mAltura)


image = Image.open('source.jpg')

w, h = 500, 375

image = conversion(initMosaico(image,w, h,5))

height, width = image.shape

bloques = piezas(image,w,h)


blanco = np.empty((height, width), dtype=np.uint8)

blanco = Image.fromarray(blanco)

for i in range(len(bloques)):
    bloques[i] = Image.fromarray(bloques[i])




coordenadas = []
for x in range(0,width, w):
    for y in range(0, height, h):
        coordenadas.append((x,y,x+w,y+h))

for i in range(len(bloques)):
    blanco.paste(bloques[i],(coordenadas[i]))



#for x in range(0,width,w):
   # for y in range(0, height,h):
  #      blanco.paste(partes[0],((x,y,x+w,y+h)))


print(type(blanco))

plt.imshow(blanco)
plt.show()