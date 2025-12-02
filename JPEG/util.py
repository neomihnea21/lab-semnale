import numpy as np

MAX_COLOR = 255

#da, sunt numere magice, dar cu astea merge conversia in ycbcr
YPBPR = np.array([
        [ 0.299,     0.587,     0.114    ],  
        [-0.168736, -0.331264,  0.5      ],  
        [ 0.5,      -0.418688, -0.081312 ]   
    ], dtype=np.float32)


def convert_pixel(r, g, b):
  r /= MAX_COLOR
  g /= MAX_COLOR
  b /= MAX_COLOR
  return np.matmul(YPBPR, np.array([r, g, b]))

def RGB_to_YCbCr(image):
  print("3 culori")
  print(np.shape(image))
  return image