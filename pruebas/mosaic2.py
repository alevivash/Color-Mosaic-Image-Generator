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

def unir(lista_de_miniaturas,w_miniatura,h_miniatura):

    for i in range(len(lista_de_miniaturas)):
        lista_de_miniaturas[i] = np.array(lista_de_miniaturas[i])

    for i in range(len(lista_de_miniaturas)):
        lista_de_miniaturas[i] = Image.fromarray(lista_de_miniaturas[i])

    p = int(round(math.sqrt(len(lista_de_miniaturas))))



    blanco = Image.new(size = (w * p, h * p), mode ="L")

    coordenadas = []
    for x in range(0,blanco.size[0],w):
        for y in range(0, blanco.size[1], h):
            coordenadas.append((x,y,x+w_miniatura,y+h_miniatura))

    for i in range(len(lista_de_miniaturas)):
        blanco.paste(lista_de_miniaturas[i],(coordenadas[i]))

    return np.array(blanco)


image = Image.open('source.jpg')

w, h, p = 500, 370, 4

image = conversion(initMosaico(image,w, h,p))

#height, width = image.shape

bloques = piezas(image,w,h)

print(len(bloques))





print(type(bloques[0]))

print(type(bloques))



regreso = unir(bloques,w,h)



plt.imshow(regreso)
plt.show()
