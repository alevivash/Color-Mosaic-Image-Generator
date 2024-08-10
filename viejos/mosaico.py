# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 00:14:57 2021

@author: Alejandro Gabriel VIVAS Hernández
"""

import math
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image


def cargar_imagenes(route):
    """
    Carga imágenes en una carpeta, las convierte en matrices numpy
    y las devuelve en una lista.

    route -> dirección absoluta donde se encuentran las imágenes.
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
    print("Se cargaron ", len(imagenes), "imagenes")
    return imagenes


def conversion(a):
    """
    Convierte un array mayor a dos dimensiones en un array de dos dimensiones.
    Esto hace que el canal RGB se anule.
    Si la matriz ya tiene dos dimensiones, devuelve el mismo array.
    """
    if np.array(a).ndim < 3:
        return np.array(a)
    else:
        return np.mean(a, axis=2, dtype=np.uint8)


def vecinoProximo(A, w, h):
    """
    Redimensiona la imagen A a las nuevas dimensiones w y h usando el método de vecino más próximo.
    """
    altura, ancho = A.shape[:2]  # dimensiones imagen
    new_image = [[A[int(altura * y / h)][int(ancho * x / w)]
                  for x in range(w)] for y in range(h)]
    return np.array(new_image)


def mediaLocal(A, w, h):
    """
    Reduce una imagen utilizando la media local de bloques.
    """
    a = np.empty((h, w), dtype=np.uint8)
    H, W = A.shape

    ph, pw = H // h, W // w
    for I in range(0, H, ph):
        for J in range(0, W, pw):
            a[I // ph, J // pw] = round(np.mean(A[I:I + ph, J:J + pw]))
    return a


def listaRGBtoGris(A):
    """
    Convierte una lista de imágenes en formato RGB a escala de grises.
    """
    for i in range(len(A)):
        A[i] = conversion(A[i])
    return A


def listaRedim(A, w, h):
    """
    Redimensiona una lista de imágenes al nuevo tamaño w x h.
    """
    for i in range(len(A)):
        A[i] = vecinoProximo(A[i], w, h)
    return A


def initMosaico(source, w, h, p):
    """
    Inicializa el mosaico redimensionando la imagen source.
    """
    mAltura, mAncho = h * p, w * p
    return vecinoProximo(source, mAncho, mAltura)


def L1(a, b):
    """
    Calcula la diferencia absoluta promedio entre dos matrices.
    """
    return abs(np.mean(a) - np.mean(b))


def escogerMiniatura(bloque, miniaturas):
    """
    Compara un bloque de una imagen con una lista de miniaturas y selecciona la más parecida.
    """
    candidatos = [L1(bloque, miniaturas[i]) for i in range(len(miniaturas))]
    minimo = min(candidatos)
    index = candidatos.index(minimo)
    return index


def unir(lista_de_miniaturas, w_miniatura, h_miniatura):
    """
    Une una lista de imágenes en un mosaico.
    """
    for i in range(len(lista_de_miniaturas)):
        lista_de_miniaturas[i] = Image.fromarray(lista_de_miniaturas[i])

    p = int(round(math.sqrt(len(lista_de_miniaturas))))
    blanco = Image.new(size=(w_miniatura * p, h_miniatura * p), mode="L")
    coordenadas = [(x, y) for x in range(0, blanco.size[0], w_miniatura)
                   for y in range(0, blanco.size[1], h_miniatura)]

    for i in range(len(lista_de_miniaturas)):
        blanco.paste(lista_de_miniaturas[i], (coordenadas[i]))

    return np.array(blanco, dtype=np.uint8)


def piezas(image, w, h):
    """
    Corta una imagen en partes según las dimensiones especificadas.
    """
    image = Image.fromarray(image)
    width, height = image.size
    partes = [(x, y, x + w, y + h) for x in range(0, width, w) for y in range(0, height, h)]
    return [np.array(image.crop(parte)) for parte in partes]


def construirMosaico(source, miniaturas, p):
    """
    Construye un mosaico a partir de una imagen base y una lista de miniaturas.
    """
    source = np.array(source)
    miniaturas = [np.array(miniatura) for miniatura in miniaturas]
    miniaturas_en_gris = listaRGBtoGris(miniaturas)
    h, w = miniaturas[0].shape[:2]
    source = conversion(initMosaico(source, w, h, p))
    bloques = piezas(source, w, h)
    area = pow(p, 2)
    indices = [escogerMiniatura(bloque, miniaturas_en_gris) for bloque in bloques]
    elegidas = [miniaturas_en_gris[i] for i in indices]
    elegidas = [Image.fromarray(elegida) for elegida in elegidas]
    mosaico = unir(elegidas, w, h)
    return np.array(mosaico, dtype=np.uint8)


# Ruta de la imagen y carga de miniaturas
imagen = Image.open('don ramon.jpg')
miniaturas = listaRedim(cargar_imagenes(r"C:\Projects\USB\Python\Mosaico\imagenes"), 30, 20)

# Construcción del mosaico
Mosaico = construirMosaico(imagen, miniaturas, 200)

# Guardar y mostrar el mosaico
plt.imsave("ejemploPara4.jpg", Mosaico, cmap='gray')
print(Mosaico.shape)
plt.imshow(Mosaico, cmap='plasma')
plt.show()

## PRIMER PROTOTIPO
