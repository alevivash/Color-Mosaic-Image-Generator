import imageio as img
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import math

def cargar_imagenes(route):
    """
    carga imagenes en una carpeta, la cual transformara en matrices numpy
    y los devolvera en una lista
    route -> direccion absoluta donde se encuentran las imagenes
    images -> lista de matrices de numpy ndarray
    """
    imagenes = []
    directory = os.fsencode(route)
    os.chdir(route)
    for file in os.listdir(directory):
        filename  = os.fsdecode(file)
        if filename.endswith(".jpg"):
            img = Image.open(filename)
            img.load()
            data = np.asarray(img,dtype="uint8")
            print(data.shape)
            imagenes.append(data)
            continue
        else:
            continue
    print("Se cargaron ", len(imagenes), "imagenes")
    return imagenes

def listaRGBtoGris(A):

    for i in range (len(A)):
        A[i] = conversion(A[i])

    return A


def listaRedim(A:[], h, w):

    for i in range (len(A)):

        A[i] = vecinoProximo(A[i], w, h)

    return A


def L1(a,b):

    return abs(np.mean(a) - np.mean(b))

def conversion(image):

    if np.array(image).ndim  < 3:
        return np.array(image)
    else:
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

def initMosaico(source, w, h, p):


    mAltura, mAncho = h * p , w * p


    return vecinoProximo(source,mAncho, mAltura)

def unir(lista_de_miniaturas,w_miniatura,h_miniatura):

    for i in range(len(lista_de_miniaturas)):
        lista_de_miniaturas[i] = np.array(lista_de_miniaturas[i])

    for i in range(len(lista_de_miniaturas)):
        lista_de_miniaturas[i] = Image.fromarray(lista_de_miniaturas[i])

    p = int(round(math.sqrt(len(lista_de_miniaturas))))

    blanco = Image.new(size = (w_miniatura* p, h_miniatura * p), mode ="L")

    coordenadas = []
    for x in range(0,blanco.size[0],w_miniatura):
        for y in range(0, blanco.size[1], h_miniatura):
            coordenadas.append((x,y,x+w_miniatura,y+h_miniatura))

    for i in range(len(lista_de_miniaturas)):
        blanco.paste(lista_de_miniaturas[i],(coordenadas[i]))

    return np.array(blanco)

def escogerMiniatura(bloque, miniaturas):

    candidatos = []
    for i in range(len(miniaturas)):

        candidatos.append(L1(bloque, miniaturas[i]))

    minimo = min(candidatos)
    index = candidatos.index(minimo)

    return index


def construirMosaico(source, miniaturas, p):

    source = np.array(source)

    for i in range (len(miniaturas)):
        miniaturas[i] = np.array(miniaturas[i])

    #source = conversion(np.array(source))

    miniaturas_en_gris = listaRGBtoGris(miniaturas)

    ejemplar = miniaturas[0]

    h,w = ejemplar.shape

    source = conversion(initMosaico(source,w,h,p))

    Height,Widht = source.shape #dimensiones ssource

    bloques = piezas(source,w,h)

    area = pow(p,2)
    indices = []
    for i in range(area):
        indices.append((escogerMiniatura(bloques[i], miniaturas_en_gris)))

    elegidas = []
    for i in indices:
        elegidas.append(miniaturas_en_gris[i])

    for i in range(area):
        elegidas[i] = Image.fromarray(elegidas[i])

    mosaico = unir(elegidas, w, h)

    return mosaico


imagen = Image.open('source.jpg')



#h,w = 88,250
p = 6


miniaturas = listaRedim(cargar_imagenes(b"C:\Users\user\Desktop\Proyecto2\imagenes"),75,100)

Mosaico = construirMosaico(imagen,miniaturas, 10)

plt.imshow(Mosaico)
plt.show()


