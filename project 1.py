import numpy as np
from PIL import Image
img = Image.open("img.jpg")
image_arr = np.array(img)
print(image_arr.shape)
print(image_arr.dtype)
print(image_arr[100,100])
grey = (0.299*image_arr[:,:,0]+0.587*image_arr[:,:,1]+0.114*image_arr[:,:,2])
grey = grey.astype(np.uint8)
Image.fromarray(grey).show()
negative = 255 - image_arr
negative_image = Image.fromarray(negative)
negative_image.show()
Gx_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

Gy_kernel = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

height, width = grey.shape
edges = np.zeros_like(grey)

# Apply convolution
for i in range(1, height-1):
    for j in range(1, width-1):

        region = grey[i-1:i+2, j-1:j+2]

        Gx = np.sum(region * Gx_kernel)
        Gy = np.sum(region * Gy_kernel)

        magnitude = np.sqrt(Gx**2 + Gy**2)

        edges[i,j] = magnitude 

# Normalize values (important)
edges = np.clip(edges, 0, 255).astype(np.uint8)

Image.fromarray(edges).show()