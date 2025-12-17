import numpy as np
from scipy import datasets
from scipy.fft import dctn, idctn
import util, copy
import matplotlib.pyplot as plt

QUANT = np.array(
         [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]])

MAGIC = 8


# we clip the right and bottom to ensure we have a whole number of blocks
def encode_layer(image):
    w = MAGIC* (np.shape(image)[1] // MAGIC) 
    h = MAGIC* (np.shape(image)[0] // MAGIC)
    canvas = np.zeros((h, w)) 
    for i in range(0, h, MAGIC):
        for j in range(0, w, MAGIC):
            block = image[i:i+MAGIC, j:j+MAGIC]
            block = QUANT * np.round(block / QUANT)
            converted_block = dctn(block)
            canvas[i:i+MAGIC, j:j+MAGIC] = converted_block
    return canvas


def decode_layer(image):
    w = MAGIC* (np.shape(image)[1] // MAGIC) 
    h = MAGIC* (np.shape(image)[0] // MAGIC)
    canvas = np.zeros((h, w)) 
    for i in range(0, h, MAGIC):
        for j in range(0, w, MAGIC):
            block = image[i:i+MAGIC, j:j+MAGIC]
            converted_block = idctn(block)
            canvas[i:i+MAGIC, j:j+MAGIC] = converted_block
    return canvas


face = datasets.face(gray=False)



