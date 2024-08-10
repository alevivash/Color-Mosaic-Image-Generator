import imageio as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Abrir imagen
image = Image.open('source.jpg')

P = np.array([[255],[212],[45]], dtype=np.uint8) #color rgb




art = np.zeros([5,5,3], dtype=np.uint8)

for x in range (5):
    for y in range(5):
        art[x,y,0] = 233
        art[x,y,1] = 126
        art[x,y,2] = 23

art[2,3,1] = 44

print((art))

plt.figure()
plt.subplot(1,2,1)
plt.imshow(P)
plt.subplot(2,2,2)
plt.imshow(art)
plt.show()