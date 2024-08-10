import imageio as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


#Este archivo es para verificar como puedo  hacer un filtro gris sin hacer uso de PIL

# Abrir imagen
imagen = Image.open('source.jpg')

# Filter
def gris_filter(image):


    return np.mean(imagen,axis=2,dtype=np.uint8)

print(type(gris_filter(imagen)))

print(gris_filter(imagen)[23,35])



# Mostrar con plt
plt.subplot()
plt.imshow(gris_filter(imagen), cmap = 'gray')
plt.show()

