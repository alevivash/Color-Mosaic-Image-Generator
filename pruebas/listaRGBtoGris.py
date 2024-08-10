import math
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def conversion(a):

    #TOMAR IMAGEN COMO UN ARRAY
    a = np.array(a)
    dimensiones = a.shape
                    #ANCHO DE LA IMAGEN
    for x in range(dimensiones[0]):
                    #ALTURA DE LA IMAGEN
        for y in range(dimensiones[1]):

            #Colores RGB
            rojo, verde, azul = a[x, y, 0],  a[x, y, 1],  a[x, y, 2]

            gris = (int (rojo) + int (verde) + (azul)) / 3

            #MEDIA GRIS
            for i in range(3):

                a[x, y, i] = gris
    return a

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



def listaRGBtoGris(A: [np.ndarray]):

    for i in range (len(A)):
        A[i] = conversion(A[i])

    return A


route = b"C:\Users\user\Desktop\Proyecto2\imagenes"

a = cargar_imagenes(route)

gris = listaRGBtoGris(a)

plt.imshow(gris[0])
plt.show()



#Ruta = directory = os.fsencode("C:\Users\user\Desktop\Proyecto2\imagenes")
#os.chdir('C:\Users\user\Desktop\Proyecto2\imagenes')