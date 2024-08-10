import imageio as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


#Como puedo modificar el color de un pixel en específico


im = Image.open('small.png') # Can be many different formats.


#1.Con PIL Image

"""
pix = im.load()

pix[5,5] = (0,0,0)
"""

#pix[x,y] = value  # Set the RGBA Value of the image (tuple)
#im.save('alive_parrot.png')  # Save the modified pixels as .png


#2.Con numpay

"""
im = np.array(im, dtype=np.uint8)

im[5,5,0] = 0
im[5,5,1] = 0
im[5,5,2] = 0


"""

plt.imshow(im)
plt.show()


"""
NOTAS:

Image.load() es una función de PIL Image la cual carga los pixeles de la imagen como si fuese una matriz de dos dimensiones. Ejm P[X,Y]=RGB

Image.size() es una función de Pil Image la cual da en términos de arrays np.shape(). Osea, da la altura y el ancho de la imagen (filas y columnas)

np.size() es una función de Numpay la cual da el número de elementos que hay en la matriz y el tamanno de dichos elementos, en el caso de una imagen depende del dtype, que establece los bits como
int8 que son 8 bits, los cuales son igual a 2bytes 
"""