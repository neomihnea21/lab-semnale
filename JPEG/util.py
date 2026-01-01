import numpy as np

MAX_COLOR = 255

#da, sunt numere magice, dar cu astea merge conversia in ycbcr
YCBCR = np.array([
        [ 0.299,     0.587,     0.114    ],  
        [-0.168736, -0.3313,  0.5      ],  
        [ 0.5,      -0.418688, -0.081312 ]   
    ])
REVERSE = np.linalg.inv(YCBCR)

def RGB_to_YCbCr(r, g, b):
   array = np.array([r, g, b]).reshape(-1, 1)
   ans = np.matmul(YCBCR.T, array)
   ans[1] += (MAX_COLOR//2 + 1)
   ans[2] += (MAX_COLOR//2 + 1)
   return ans

def YCbCr_to_RGB(y, cb, cr):
   array = np.array([y, cb, cr]).reshape(-1, 1)
   array[1] -= (MAX_COLOR//2 + 1)
   array[2] -= (MAX_COLOR//2 + 1)
   
   ans = np.matmul(REVERSE.T, array)
   return np.clip(ans, 0, MAX_COLOR).astype(np.uint8)

def convert_colorised_image(image):
   L = np.shape(image)[0]
   W = np.shape(image)[1]

   modified_face = np.zeros_like(image)
   for i in range(L):
     for j in range(W):
        r, g, b = image[i, j, 0], image[i, j, 1], image[i, j, 2]
        converted_pixel = RGB_to_YCbCr(r, g, b)
        for k in range(3):
           modified_face[i, j, k] = converted_pixel[k]
   return modified_face

def revert_colorised_image(image):
   L = np.shape(image)[0]
   W = np.shape(image)[1]

   modified_face = np.zeros_like(image)
   for i in range(L):
     for j in range(W):
        r, g, b = image[i, j, 0], image[i, j, 1], image[i, j, 2]
        converted_pixel = YCbCr_to_RGB(r, g, b)
        for k in range(3):
           modified_face[i, j, k] = converted_pixel[k]
   return modified_face