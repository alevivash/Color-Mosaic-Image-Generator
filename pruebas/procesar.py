import imageio as img
import matplotlib.pyplot as plt
import numpy as np

source = img.imread ('source.jpg')
T = np.shape(source)

print(T)

print(T[0])
negativo = np.zeros(T)


for i in range(T[0]): #Altura
   for j in range(T[1]): #Ancho
        negativo[i,j,0] = 255 - source[i,j,0] #r
        negativo[i,j,1] = 255 - source[i,j,1] #g
        negativo[i,j,2] = 255 - source[i,j,2] #b

negativo = np.uint8(negativo)

print(type(negativo))

plt.imshow(negativo)
plt.show()