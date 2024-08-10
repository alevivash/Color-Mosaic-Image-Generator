import imageio as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

"""

def reduccionSumas1(A:np.ndarray,S:np.ndarray,w,h):
    a = np.empty((h,w), dtype=np.uint8)
    H,W = A.shape
    ph, pw = H // h, W // w
    nbp = ph * pw
    for I in range (0,W, ph):
        for J in range (0, H, pw):
            X = (S[I+ph, J+pw] - S[I+ph, J]) - (S[I,J+pw] - S[I,J])
            a[I//ph, J//pw] = round(X / nbp)

    return a
"""


def conversion(image):

    return np.mean(image,axis=2,dtype=np.uint8)

def reduccionSumas2(A:np.ndarray,S:np.ndarray,w,h):
    #a = np.empty((h,w), dtype=np.uint8)
    H,W = A.shape
    ph, pw = H // h, W // w
    sred = S[0:H+1:ph, 0:W+1:pw]
    dc = sred[:, 1:] - sred[:,:-1]
    d1 = dc[1:, :] - dc[:-1, :]
    d =  d1/ (ph * pw)
    return  np.uint8(d.round())


def tablaSuma(A):

    A = np.array(A, dtype=np.uint8)

    B = np.empty(A.shape)

    B.resize((B.shape[0]+1, B.shape[1]+1))

    return B





#Abrir imagen
image = Image.open('source.jpg')
image = np.array(image)

gris = conversion(image)


tabla = tablaSuma(gris)

print(tabla.shape)


new_image = reduccionSumas2(gris, tabla, 1250 ,625 )

#print(new_image.shape)

print(new_image.shape)


plt.imshow(new_image)
plt.show()