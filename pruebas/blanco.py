import imageio as imge
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


img = np.ones([1,1,3], dtype=np.uint8)*255


print(type(img))

print(img)

print(img.shape)
print(img.size)


#plt.imshow(img)
#plt.show()