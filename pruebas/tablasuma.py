import imageio as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



def conversion(image):

    return np.mean(image,axis=2,dtype=np.uint8)

def tablaSuma(A):

    A = np.array(A, dtype=np.uint8)

    B = np.array(A, dtype= int)

    B.resize((B.shape[0] + 1, B.shape[1] + 1))

    return B

def tablaSuma2(A):

    A = np.array(A, dtype=np.uint8)

    B = np.empty((A.shape[0] + 1, A.shape[1] + 1))

    return B

def reduccionSumas2(A:np.ndarray,S:np.ndarray,w,h):
    #a = np.empty((h,w), dtype=np.uint8)
    H,W = A.shape
    ph, pw = H // h, W // w
    sred = S[0:H+1:ph, 0:W+1:pw]
    dc = sred[:, 1:] - sred[:,:-1]
    d1 = dc[1:, :] - dc[:-1, :]
    d =  d1/ (ph * pw)
    return  np.uint8(d.round())


small = Image.open('small.png')

source = Image.open('source.jpg')

small = conversion(small)

source = conversion(source)

tabla = tablaSuma(small)
tabla2 = tablaSuma2(small)

print(tabla.shape)

reducidasmall = reduccionSumas2(small,tabla,12,12)


tablaS = tablaSuma(source)
tablaS2 = tablaSuma2(source)

print(tablaS2.dtype)

print(tablaS.shape)

reducidasource = reduccionSumas2(source, tablaS2, 1250, 625)


plt.imshow(tablaS)
plt.show()