import os
import numpy as np
from PIL import Image
import math

def cargar_imagenes(route):

    """
    Carga imagenes en una carpeta, la cual transformara en matrices numpy
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

def piezas(image, w, h):
    """
    Esta función se encarga de cortar una imagen (np.array) en partes, si las dimensiones de la imagenes miniaturas son divisibles entre las dimensiones de la imagen source.
    Devuelve una lista de imágenes con las partes
    """

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
    """
    Este código se encarga de unir una lista de imagenes de w_miniatura de ancho, por h_miniaturas de alto.
    """
    ##Convierte tu lista de np.array de imagenes a formato PIL (image.fromarray)
    for i in range(len(lista_de_miniaturas)):
        lista_de_miniaturas[i] = Image.fromarray(lista_de_miniaturas[i])

    p = int(round(math.sqrt(len(lista_de_miniaturas)))) ##cantidad de cuadricula por lado o pixeles

    # Crea una base en donde pegar las images
    blanco = Image.new(size=(w_miniatura * p, h_miniatura * p), mode="RGB")

    coordenadas = []
    for x in range(0, blanco.size[0], w_miniatura):
        for y in range(0, blanco.size[1], h_miniatura):
            coordenadas.append((x, y, x + w_miniatura, y + h_miniatura))

    # Pega las imagenes en la base
    for i in range(len(lista_de_miniaturas)):
        blanco.paste(lista_de_miniaturas[i], (coordenadas[i][0], coordenadas[i][1]))

    return np.array(blanco, dtype=np.uint8)

def escogerMiniatura(bloque, lista_miniaturas):
    """
    Compara un bloque de una imagen source con una lista de imagenes,
    usando la función L1 para obtener el valor más parecido (mínima diferencia)
    """
    candidatos = [L1(bloque, miniatura) for miniatura in lista_miniaturas]
    index = np.argmin(candidatos)
    return index

def listaRedim(A, w, h):
    """
    Cambia la altura y el ancho de una lista de imágenes del mismo tamaño por medio de la función vecinoProximo
    anteriormente nombrada
    """

    for i in range(len(A)):
        A[i] = vecinoProximo(A[i], w, h)

    return A

def vecinoProximo(A, w, h):
    """
    Toma las dimensiones de la imagen A y las divide entre las nuevas dimensiones dadas, w y h. A través de un bucle for
    """

    altura, ancho = A.shape[0], A.shape[1]  # dimensiones imagen

    new_image = [[A[int(altura * y / h)][int(ancho * x / w)]
                  for x in range(w)] for y in range(h)]

    return np.array(new_image)

def L1(a, b):
    """
    Toma el promedio de todos los elementos de cada matriz y retorna el grado de diferencia
    """
    return abs(np.sum(np.abs(a - b)))

def initMosaico(source, w: int, h: int, p: int):
    mAltura, mAncho = h * p, w * p
    return vecinoProximo(source, mAncho, mAltura)

def construirMosaico(source, miniaturas, p):
    """
    Construye una imagen como un mapa de bits de dos dimensiones (escala de grises) con una imagen base(source),
    una lista de miniaturas y la cantidad de miniaturas por lado (p).

    """
    source = np.array(source)

    for i in range(len(miniaturas)):
        miniaturas[i] = np.array(miniaturas[i])

    ejemplar = miniaturas[0]
    h, w, _ = ejemplar.shape

    source = initMosaico(source, w, h, p)

    # Corta en bloques la imagen original(source)
    bloques = piezas(source, w, h)

    area = pow(p, 2)

    # Compara cada bloque con todos las imagenes miniatura y selecciona el más parecido para cada bloque
    indices = []
    for i in range(area):
        indices.append(escogerMiniatura(bloques[i], miniaturas))

    elegidas = []
    for i in indices:
        elegidas.append(miniaturas[i])

    # Une las imagenes elegidas con mayor parecido y crea el mosaico de imagenes
    mosaico = unir(elegidas, w, h)

    # retorna el mosaico como un array
    return np.array(mosaico, dtype=np.uint8)

ramon = Image.open('The Joker.jpg')
miniaturas = listaRedim(cargar_imagenes(b"C:\Projects\USB\Python\Mosaico\imagenes"), 28, 42)
Mosaico = construirMosaico(ramon, miniaturas, 10)
Imagen = Image.fromarray(Mosaico)
Imagen.show()

##Esta es la que se ve peor.
# Los bloques que son muy luminosos o saturados de color son remplzadaos por la imagen mas blanca (luminence idk)