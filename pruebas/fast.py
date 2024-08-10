import imageio as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import math

def vecinoProximo(A,w,h):
    altura, ancho = A.shape[0], A.shape[1] #dimensiones imagen

    new_image = [[A[int(altura * y / h)][int(ancho * x / w)]
                     for x in range(w)] for y in range(h)]

    return np.array(new_image)


image = np.array(Image.open('normal.jpg'))

mini1, mini2 = vecinoProximo(image,500, 375), vecinoProximo(image,500, 375)

junto = np.empty((375,1000,3), dtype=np.uint8)

"""
for y in range(junto.shape[0]):
    for x1 in range(0, int(junto.shape[1] / 2)):

        junto[y,x1] = mini2[y,x1]

"""


print(mini1[23,34,1])

nuevo = np.array([[125,121,12],[123,31,223],[0,35,208]], dtype=np.uint8)

print(math.sqrt(25))


"""
plt.subplot(1,2,1)
plt.imshow(mini1)
plt.subplot(1,2,2)
plt.imshow(mini2)
plt.show()
"""

#plt.imshow(junto)
#plt.show()
