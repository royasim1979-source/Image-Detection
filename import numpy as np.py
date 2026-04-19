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
