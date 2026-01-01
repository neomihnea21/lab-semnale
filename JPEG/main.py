import numpy as np
import matplotlib.pyplot as plt
import scipy.datasets
import sys, skimage, scipy
import encode, huffman, video, util
def ex1(quality_factor=1):
   image = scipy.datasets.ascent()
   print(np.shape(image))
   new_image = encode.encode_layer(image, quality_factor)
   restored_image = encode.decode_layer(new_image)
   plt.imsave("img/buidings_coded.jpg", restored_image)

# this one has a 3d image
def ex2():
   image = util.convert_colorised_image(scipy.datasets.face())
   print(np.shape(image))
   coded_image = np.dstack((encode.encode_layer(image[:, :, 0]), encode.encode_layer(image[:, :, 1]), encode.encode_layer(image[:, :, 2])))
   restored_image = np.dstack((encode.decode_layer(coded_image[:, :, 0]), encode.decode_layer(coded_image[:, :, 1]), encode.decode_layer(coded_image[:, :, 2])))
   print(f"DECODAT: {np.shape(restored_image)}")
   final = util.revert_colorised_image(restored_image)
   plt.imsave("img/face.jpg", final / 255)

def disk_save(path, quality_factor = 1):
    image = skimage.io.imread(path)
    image = util.convert_colorised_image(image)
    coded_image = np.dstack((encode.encode_layer(image[:, :, 0], quality_factor),
                             encode.encode_layer(image[:, :, 1], quality_factor),
                             encode.encode_layer(image[:, :, 2], quality_factor)))

def ex3(path, quality_factor = 1):
    # assume it's color
   
    
def main():
    ex2()
            

if __name__ == "__main__":
    main()
