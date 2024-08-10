import imageio as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Abrir imagen
image = Image.open('source.jpg')
image = np.array(image, dtype=np.uint8)

#o = np.ones((16,16))

#print(o.shape)

vacio = np.empty((image.shape[0], image.shape[1], 3), dtype=np.uint8)
print(vacio.shape)

print(image[121,323,0])
print(image[121,323,1])
print(image[121,323,2])

#vacio = np.mean()

def conversion(image):

    return np.mean(image,axis=2,dtype=np.uint8)


def medialocal(A: np.ndarray, w, h):
    a = np.empty((h,w), dtype= np.uint8)
    H,W = A.shape

    ph,pw = H//h, W//w
    for I in range(0,H,ph):
        for J in range(0,W,pw):
            a[I//ph,J//pw,]=round(np.mean(A[I:I+ph, J:J+pw]))


    return a

gris = conversion(image)

new_image = medialocal(gris,1250,625)

#los datos que puedo usar en w y en h son los números en que son divisibles dichos valores

#plt.figure()
#plt.subplot(1,2,1)
#plt.imshow(new_image, cmap='plasma')
#plt.subplot(2,2,2)
plt.imshow(new_image)
plt.show()
