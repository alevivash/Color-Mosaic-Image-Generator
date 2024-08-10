import imageio as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def most_frequent_color(image):
    w, h = image.size
    pixels = image.getcolors(w * h)

    most_frequent_pixel = pixels[0]
    for count, color in pixels:

        if count > most_frequent_pixel[0]:
            most_frequent_pixel = color

        if len(most_frequent_pixel) == 2:
            return most_frequent_pixel[1]
        else:
            return most_frequent_pixel

def conversion(a):

    return np.mean(a,axis=2,dtype=np.uint8)

def frecuente (imagen):

    return np.mean(imagen)



imagen = (conversion(Image.open('azul.jpg')))


print('el valor más frecuente es ' + str(np.mean(imagen)))



#imagen = Image.fromarray(imagen)

#frequent = most_frequent_color(imagen)

#plt.imshow(imagen)
#plt.show()





