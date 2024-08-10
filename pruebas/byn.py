#def grayscale(int height, int width, RGBTRIPLE image[height][width])
import imageio as img
import matplotlib.pyplot as plt
import numpy as np

A = np.zeros((20,20))

for i in range(len(A)):
    A[i,i] = True
    A[i, len(A) - i - 1] = True


print (len(A))


plt.imshow(A, cmap = 'plasma')
plt.show()

