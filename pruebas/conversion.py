import imageio as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Abrir imagen
image = Image.open('source.jpg')


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

new = conversion(image)

plt.imshow(new)
plt.show()



