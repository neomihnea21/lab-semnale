import numpy as np

MAX_COLOR = 255

#da, sunt numere magice, dar cu astea merge conversia in ycbcr
YCBCR = np.array([
        [ 0.299,     0.587,     0.114    ],  
        [-0.168736, -0.331264,  0.5      ],  
        [ 0.5,      -0.418688, -0.081312 ]   
    ], dtype=np.float32)


def RGB_to_YCbCr(image):
   image /= MAX_COLOR # normalization
   ycbcr_image = image @ YCBCR.T
   Y_layer, Cb_layer, Cr_layer = np.split(ycbcr_image, 3, axis=-1)
   return np.round(Y_layer.squeeze()*MAX_COLOR-128), np.round(Cb_layer.squeeze()*MAX_COLOR), np.round(Cr_layer.squeeze()*MAX_COLOR)

def YCbCr_to_RGB(image):
   image[:, :, 0] += 128
   image /= 255
   RGB_normalized = image @ np.linalg.inv(YCBCR).T
   RGB_real = np.round(RGB_normalized*MAX_COLOR).astype(np.uint8)
   return RGB_real