import imageio as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Abrir imagen
image = Image.open('source.jpg')

# Filter
def gris_filter(image):
    #Dimensiones de la imagen
    width, height = image.size

    # Crer una nueva imagen con formato RGB
    new_image = Image.new("RGB", (width, height), "white")

    #Filtro
    for x in range(width): #2500
        for y in range(height): #1875
            # Get original pixel colors
            r, g, b = image.getpixel((x, y))

            # New pixel colors
            r_ = g_ = b_ = (r + g + b) / 3

            # Change new pixel
            new_pixel = (int(r_), int(g_), int(b_))
            new_image.putpixel((x, y), new_pixel)

    return new_image

# Mostrar con plt
plt.imshow(gris_filter(image))
plt.show()