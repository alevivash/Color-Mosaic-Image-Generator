# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 00:14:57 2021

@author: Alejadro Gabriel VIVAS Hernández
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import numpy as np
from PIL import Image
import cv2


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
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            img = Image.open(filename)
            img.load()
            data = np.asarray(img, dtype="uint8")
            print(data.shape)
            imagenes.append(data)
            continue
        else:
            continue
    print("Se cargaron ", len(imagenes), "imagenes")
    return imagenes


def gris(p):
    pass


def conversion(a):
    """
    Esta función convierte un array mayor a dos dimensiones en un array de dos dimensiones. Esto hace que el canal RGB se anule
    En tal caso la matriz ya tenga  dos dimensiones, devuelve el mismo array.
    """

    if np.array(a).ndim < 3:
        return np.array(a)
    else:
        return np.mean(a, axis=2, dtype=np.uint8)

    pass


def vecinoProximo(A, w, h):
    """
    Toma las dimensiones de la imagen A y las divide entre las nuevas dimensiones dadas, w y h. A través de un bucle for
    """

    altura, ancho = A.shape[0], A.shape[1]  # dimensiones imagen

    new_image = [[A[int(altura * y / h)][int(ancho * x / w)]
                  for x in range(w)] for y in range(h)]

    return np.array(new_image)

    pass


def mediaLocal(A, w, h):
    """
    Este código es útil para reducir una imagen, funciona de la siguiente forma:
    1-Toma los parametros de dimensiones de la imagen A. H (Altura) y W (Ancho)
    2-Obtiene nuevos parametros ph y pw diviendo las dimensiones de la imagen y las dimensiones que estan como argumento (w y h).
    Estos valores marcan la secuencia en los siguientes bucles.
    3-Se itera en la filas y columas de la imagen por medio de una secuencia en la columnas la secuencia esta marcada por pw y en las filas por ph.
    4-Divide los números valores que se obtienen la iteracion (I y J) entre los valores que marcan la secuencia,
    5-Para que funcione el código es necesario, colocar como argumento w y h, enteros que sean divisibles entre las dimensiones originales de la imagen
    H y W
    :param A:
    :param w:
    :param h:
    :return:
    """

    a = np.empty((h, w), dtype=np.uint8)
    H, W = A.shape

    ph, pw = H // h, W // w
    for I in range(0, H, ph):
        for J in range(0, W, pw):
            a[I // ph, J // pw,] = round(np.mean(A[I:I + ph, J:J + pw]))
    return a
    pass


def TablaSuma(A):
    pass


def reduccionSumas1(A, S: np.ndarray, w: int, h: int):
    pass


def reduccionSumas2(A, S: np.ndarray, w: int, h: int):
    pass


def listaRGBtoGris(A):
    """
     Convierte una lista de imágenes en formato RGB a una tabla de pixeles de dos dimensiones por medio la función conversión
     anteriormente nombrada
    """

    for i in range(len(A)):
        A[i] = conversion(A[i])

    return A

    pass


def listaRedim(A, w, h):
    """
    Cambia la altura y el ancho de una lista de imágenes del mismo tamaño por medio de la función vecinoProximo
    anteriormente nombrada
    """

    for i in range(len(A)):
        A[i] = vecinoProximo(A[i], w, h)

    return A

    pass


def initMosaico(source, w: int, h: int, p: int):
    mAltura, mAncho = h * p, w * p

    return vecinoProximo(source, mAncho, mAltura)
    pass


def L1(a, b):
    """
    Toma el promedio de todos los elementos de cada matriz y luego obtiene la diferencia de estos
    """

    return abs(np.mean(a) - np.mean(b))
    pass


def escogerMiniatura(bloque: np.ndarray, miniaturas):
    """
    Compara un bloque de una imagen source con una lista de imagenes, con ayuda de la funcion L1, obtiene el valor más parecido (Mínima diferencia) y lo devuelve
    """

    candidatos = []
    for i in range(len(miniaturas)):
        candidatos.append(L1(bloque, miniaturas[i]))

    minimo = min(candidatos)
    index = candidatos.index(minimo)

    return index

    pass


def unir(lista_de_miniaturas, w_miniatura, h_miniatura):
    """
    Este código se encarga de unir una lista de imagenes de w_miniatura de ancho, por h_miniaturas de alto.
    """

    ##En tal caso la imagen este en formato Pil, la convierte en np.array
    for i in range(len(lista_de_miniaturas)):
        lista_de_miniaturas[i] = np.array(lista_de_miniaturas[i])

    for i in range(len(lista_de_miniaturas)):
        lista_de_miniaturas[i] = Image.fromarray(lista_de_miniaturas[i])

    p = int(round(math.sqrt(len(lista_de_miniaturas))))

    # Crea una base en donde pegar las images
    blanco = Image.new(size=(w_miniatura * p, h_miniatura * p), mode="RGB")

    coordenadas = []
    for x in range(0, blanco.size[0], w_miniatura):
        for y in range(0, blanco.size[1], h_miniatura):
            coordenadas.append((x, y, x + w_miniatura, y + h_miniatura))

    # Pega las imagenes en la base
    for i in range(len(lista_de_miniaturas)):
        blanco.paste(lista_de_miniaturas[i], (coordenadas[i]))

    return np.array(blanco, dtype=np.uint8)


def piezas(image, w, h):  # --->image: lista de imagenes en partes, h y w dimensiones imagenes miniaturas(piezas)

    """
    Esta función se encarga de cortar una imagen en partes, si las dimensiones de la imagenes miniaturas son divisibles entre las dimensiones de la imagen source.
    Devuelve una lista de imágenes con las partes
    """

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


def construirMosaico(source, miniaturas, p):
    """
    Construye una imagen como un mapa de bits de dos dimensiones (escala de grises) con una imagen base(source), una lista de miniaturas y la cantidad de miniaturas por lado (p).

    """

    source = np.array(source)

    for i in range(len(miniaturas)):
        miniaturas[i] = np.array(miniaturas[i])

    ejemplar = miniaturas[0]

    h, w = ejemplar.shape[0], ejemplar.shape[1]

    source = (initMosaico(source, w, h, p))

    bloques = piezas(source, w, h)

    area = pow(p, 2)

    # Compara cada bloque con todos las imagenes miniatura y selecciona el más parecido para cada bloque
    indices = []
    for i in range(area):
        indices.append((escogerMiniatura(bloques[i], miniaturas)))

    elegidas = []
    for i in indices:
        elegidas.append(miniaturas[i])

    for i in range(area):
        elegidas[i] = Image.fromarray(elegidas[i])

    # Une las imagenes elegidas con mayor parecido y crea el mosaico de imagenes

    mosaico = unir(elegidas, w, h)

    # retorna el mosaico como un array
    return np.array(mosaico, dtype=np.uint8)

    pass


def construirMosaicogris(source, miniaturas, p):
    """
    Construye una imagen como un mapa de bits de dos dimensiones (escala de grises) con una imagen base(source), una lista de miniaturas y la cantidad de miniaturas por lado (p).

    """

    source = np.array(source)

    for i in range(len(miniaturas)):
        miniaturas[i] = np.array(miniaturas[i])

    # source = conversion(np.array(source))

    miniaturas_en_gris = listaRGBtoGris(miniaturas)

    ejemplar = miniaturas[0]

    h, w = ejemplar.shape

    source = conversion(initMosaico(source, w, h, p))

    Height, Widht = source.shape  # dimensiones ssource

    # Corta en bloques la imagen original(source)

    bloques = piezas(source, w, h)

    area = pow(p, 2)

    # Compara cada bloque con todos las imagenes miniatura y selecciona el más parecido para cada bloque
    indices = []
    for i in range(area):
        indices.append((escogerMiniatura(bloques[i], miniaturas_en_gris)))

    elegidas = []
    for i in indices:
        elegidas.append(miniaturas_en_gris[i])

    for i in range(area):
        elegidas[i] = Image.fromarray(elegidas[i])

    # Une las imagenes elegidas con mayor parecido y crea el mosaico de imagenes

    mosaico = unir(elegidas, w, h)

    # retorna el mosaico como un array

    return np.array(mosaico, dtype=np.uint8)

    pass


imagen = Image.open('source1.jpg')

imagen = np.array(imagen)

miniaturas = listaRedim(cargar_imagenes(b"C:\Projects\USB\Python\Mosaico\imagenes"), 100, 75)

Mosaico = construirMosaicogris(imagen, miniaturas, 50)

plt.imsave("ejemplo2", Mosaico, cmap='gray')

# plt.imshow(Mosaico)

# plt.show()

"""
Q1: Debido a que cada componente tiene un valor entre el 0 a 250. Existen 250^3 combinaciones posibles, es decir  15,625,000 colores posibles

Q2: 
Instrucción tabla de numpay correspondiente pixel blanco
img = np.ones([1,1,3], dtype=np.uint8)*255

print(type(img))

plt.imshow(img)
plt.show()

Q3: Considerando a = np.uint8(280) y b np.uint8(240).

a = np.uint8(280) : 24 ----> Debido a que se excedió del límite de 255 valores, contados desde el principio por eso es 24
b = np.uint8(240) : 240 --> El número se encuentra en el rango de [0,255], así que no hay errores
(a+b) : 8 ----> Da como resultado 8 pero muestra un error RuntimeWarning que representa un caso de desbortamiento, siendo 8 un número
que no representa la suma real
(a-b) : 40----> Mismo resultado que con la suma anterior, mostrado un error RuntimeWarning y arrojando como resultado 40, que no es el
verdadero valor de la resta
(a / b): 0.1 ----> Es el resultado de dividir 24 entre 240. Es un valor flotante
(a // b): 0. -----> Es la aproximación de dividir 24 entre 240 como valor entero. 

Q13. Para poder hacer un fotomosaico con 50 miniaturas, se debe la imagen source a 3750x5000 mientras que las imagenes miniaturas se 
deben redimensionar a 75x100

"""