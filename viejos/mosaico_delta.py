import os
import numpy as np
from PIL import Image
import colour
import math

def cargar_imagenes(route):
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

def piezas(image, w, h):
    image = Image.fromarray(image)
    width, height = image.size
    partes = []
    for x in range(0, width, w):
        for y in range(0, height, h):
            partes.append((x, y, x + w, y + h))
    for i in range(len(partes)):
        partes[i] = image.crop(partes[i])
        partes[i] = np.array(partes[i])
    return partes

def unir(lista_de_miniaturas, w_miniatura, h_miniatura):
    for i in range(len(lista_de_miniaturas)):
        lista_de_miniaturas[i] = Image.fromarray(lista_de_miniaturas[i])
    p = int(round(math.sqrt(len(lista_de_miniaturas))))
    blanco = Image.new(size=(w_miniatura * p, h_miniatura * p), mode="RGB")
    coordenadas = []
    for x in range(0, blanco.size[0], w_miniatura):
        for y in range(0, blanco.size[1], h_miniatura):
            coordenadas.append((x, y, x + w_miniatura, y + h_miniatura))
    for i in range(len(lista_de_miniaturas)):
        blanco.paste(lista_de_miniaturas[i], (coordenadas[i][0], coordenadas[i][1]))
    return np.array(blanco, dtype=np.uint8)

def escogerMiniatura(bloque, lista_miniaturas):
    candidatos = [distance_colour(bloque, miniatura) for miniatura in lista_miniaturas]
    index = np.argmin(candidatos)
    return index

def listaRedim(A, w, h):
    for i in range(len(A)):
        A[i] = vecinoProximo(A[i], w, h)
    return A

def vecinoProximo(A, w, h):
    altura, ancho = A.shape[0], A.shape[1]
    new_image = [[A[int(altura * y / h)][int(ancho * x / w)]
                  for x in range(w)] for y in range(h)]
    return np.array(new_image)

def distance_colour(a, b):  # Lenght between 2 points luminance and color
    """
    Calcula una combinación de distancia euclidiana en el espacio de color y distancia Manhattan
    """

    Delta_E = colour.delta_E(a, b, 'CIE 1976')

    return Delta_E #np.mean(Delta_E + L1)



def initMosaico(source, w, h, p):
    mAltura, mAncho = h * p, w * p
    return vecinoProximo(source, mAncho, mAltura)

def construirMosaico(source, miniaturas, p):
    source = np.array(source)
    for i in range(len(miniaturas)):
        miniaturas[i] = np.array(miniaturas[i])
    ejemplar = miniaturas[0]
    h, w, _ = ejemplar.shape
    source = initMosaico(source, w, h, p)
    bloques = piezas(source, w, h)
    area = pow(p, 2)
    indices = [escogerMiniatura(bloque, miniaturas) for bloque in bloques]
    elegidas = [miniaturas[i] for i in indices]
    mosaico = unir(elegidas, w, h)
    return np.array(mosaico, dtype=np.uint8)

ramon = Image.open('Killua.jpg')
miniaturas = listaRedim(cargar_imagenes(b"C:\Projects\USB\Python\Mosaico\imagenes"), 14, 21)
Mosaico = construirMosaico(ramon, miniaturas, 10)
Imagen = Image.fromarray(Mosaico)
Imagen.show()