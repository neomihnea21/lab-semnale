import numpy as np
from scipy import datasets
from scipy.fft import dctn, idctn
import util
import matplotlib.pyplot as plt

QUANT = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]])

MAGIC = 8
# for now, let's assume the image size is divisible by 8 in both dimensions
def encode_layer(image):
    len_blocks = len(image[0]) // MAGIC 
    h_blocks = len(image) // MAGIC 
    for i in range(len_blocks):
        for j in range(h_blocks):
            block = image[i*MAGIC:i*MAGIC+8, j*MAGIC: j*MAGIC+8]
            transformed_block = dctn(block)
            block_jpeg = QUANT * np.round(transformed_block / QUANT)
            block_jpeg = idctn(block_jpeg)
    return image

def encode(image: np.array):
    if image.ndim > 2:
        y, Cb, Cr=util.RGB_to_YCbCr(image)
        Cb += 128
        Cr += 128
        image[:, :, 0] = y
        image[:, :, 1] = Cb
        image[:, :, 2] = Cr
        image = np.copy(image) 
        for i in range(3):
            image[i] = encode_layer(image[i])
        
    else:
        image = encode_layer(image) # there's a single layer, if we're in grayscale
    return image

def decode(image):
    if image.ndim > 2:
        image=util.YCbCr_to_RGB(image)


buildings = datasets.ascent()
print(np.shape(buildings))
image = encode(buildings)
image = decode(image)
image=image.astype(np.float32)
plt.imshow(image, cmap=plt.cm.gray)
plt.show()