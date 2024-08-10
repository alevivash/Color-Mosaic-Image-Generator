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

    return partes #---> lista de la imagen en porciones np.array


image = Image.open('source.jpg')

#image = conversion(image)

image = vecinoProximo(image,3750,2000)

w, h =  375, 200
partes = piezas(image, w, h)

#print(len(partes[90:101]))

#matriz = np.array([partes[0:10], partes[10:20], partes[20:30], partes[30:40], partes[40:50], partes[50:60], partes[60:70], partes[70:80], partes[80:90], partes[90:100]], dtype=np.uint8)
p = 10


for i in range (0,p * p,p):

    matriz = np.array([partes[i:i + p], + partes ])

matriz = np.array([partes[0:10], partes[10:20], partes[20:30], partes[30:40], partes[40:50], partes[50:60], partes[60:70], partes[70:80], partes[80:90], partes[90:100]], dtype=np.uint8)


print(matriz.shape)

print(type(partes))

#print(matriz[0,0])


#blanco = np.empty((height, width), dtype=np.uint8)

#blanco = Image.fromarray(blanco)

#verde = Image.fromarray(conversion(vecinoProximo(Image.open('verde.jpg'),w,h)))


"""

for x in range(0,width,w):
    for y in range(0, height,h):
        blanco.paste(partes[99],((x,y,x+w,y+h)))
            #for i in range(len(partes))

for i in range(len(partes)):
    blanco.paste(partes[0],((375,200 ,375 * 2, 2 * 200)))


#blanco.paste(partes[0],(0,0))
"""

plt.imshow(matriz[9,9])
plt.show()

