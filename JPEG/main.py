import numpy as np
import matplotlib.pyplot as plt
import scipy.datasets
import sys, skimage, scipy, imageio
import encode, huffman, video, util, cv2
def ex1(quality_factor=1):
   image = scipy.datasets.ascent()
   print(np.shape(image))
   new_image = encode.encode_layer(image, quality_factor)
   restored_image = encode.decode_layer(new_image)
   plt.imsave("img/buidings_coded.jpg", restored_image)

# this one has a 3d image
def ex2(image):
   image = trim(image)
   print(np.shape(image))
   coded_image = np.dstack((encode.encode_layer(image[:, :, 0]), encode.encode_layer(image[:, :, 1]), encode.encode_layer(image[:, :, 2])))
   restored_image = np.dstack((encode.decode_layer(coded_image[:, :, 0]), encode.decode_layer(coded_image[:, :, 1]), encode.decode_layer(coded_image[:, :, 2])))
   print(f"DECODAT: {np.shape(restored_image)}")
   final = util.revert_colorised_image(restored_image)
   return final / 255


def disk_save(image_path, quality_factor = 1):
    image = skimage.io.imread(image_path)
    image = util.convert_colorised_image(image)
    coded_image = np.dstack((encode.encode_layer(image[:, :, 0], quality_factor),
                             encode.encode_layer(image[:, :, 1], quality_factor),
                             encode.encode_layer(image[:, :, 2], quality_factor)))
    streams = []
    trees = []
    for k in range(3):
        flat = coded_image[:, :, k].flatten()
        tree = huffman.encode_huffman(flat)
        codebook = dict()
        huffman.get_codes(tree, "", codebook)
        stream = "".join(codebook[val] for val in flat)
        streams.append(stream)
        trees.append(tree)

    return streams, trees
        # we could have committed to disk, but we won't, for ease of use

def reload_layer(stream, image_shape, tree: huffman.HuffmanNode):
    rebuilt_numbers = huffman.decode_huffman(stream, tree)
    arr = np.array(rebuilt_numbers)
    return np.reshape(arr, image_shape)

#this allows us to compress the blackbird as far as we want
# beware, it is VERY lossy
def ex3_v1(image_path, quality_factor = 1):
    streams, trees = disk_save(image_path, quality_factor)
    image = skimage.io.imread(image_path)
    image = trim(image)
    print(f"FORMA: {np.shape(image)}")
    shape = (np.shape(image)[0], np.shape(image)[1])
    canvas = np.zeros_like(image)
    for k in range(3):
        layer = reload_layer(streams[k], shape, trees[k])
        canvas[:, :, k] = layer
    canvas = util.revert_colorised_image(canvas)
    plt.imsave("img/huffed_blackbird_2.jpg", canvas)

def mse(x, y):
    return np.mean((x-y) ** 2)

def trim(image):
    l, w, _ = np.shape(image)
    return image[:(l-l%8), :(w-w%8), :]

def ex3_aux(image_path, quality_factor):
    image = skimage.io.imread(image_path)
    image = trim(image)
    coded_image = np.dstack((encode.encode_layer(image[:, :, 0], quality_factor),
                             encode.encode_layer(image[:, :, 1], quality_factor),
                             encode.encode_layer(image[:, :, 2], quality_factor)))
    restored_image = np.dstack((encode.decode_layer(coded_image[:, :, 0]), encode.decode_layer(coded_image[:, :, 1]), encode.decode_layer(coded_image[:, :, 2])))
    final = util.revert_colorised_image(restored_image)
    print(np.shape(coded_image))
    return mse(image, final), final

# implementarea EX3 finala din tema
def ex3_v2(image_path, MSE_target):
    qual = 0
    step = 20
    while(step > 0.01):
        score, _ = ex3_aux(image_path, qual+step)
        if score < MSE_target:
            qual += step
        step /= 2
    _, coded_image = ex3_aux(image_path, qual)
    return coded_image

def ex4(video_path, write_path):
    frames = video.get_frames(video_path)
    print(np.shape(frames))
    print(np.shape(ex2))
    fps = 24 
    image_size = (np.shape(frames)[1], np.shape(frames)[2])
    with imageio.get_writer(write_path, fps=fps, codec='libx264', macro_block_size=1, format="FFMPEG") as writer:
        for frame in frames:
            parsed_frame = (ex2(frame)*255).astype(np.uint8)
            writer.append_data(parsed_frame)
def main():
    imag = ex3_v2("img/blackbird.jpg", 14700)
    plt.imsave("img/saved_blackbird.jpg", imag / 255)
            

if __name__ == "__main__":
    main()
