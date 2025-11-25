import numpy as np
import matplotlib.pyplot as plt
import scipy.datasets as data
import copy

def energy(image):
    new_image = np.ndarray.flatten(image)
    return np.dot(new_image, new_image)

img = data.face(gray=True)
img_stash = copy.deepcopy(img)

img_freq = np.fft.fft2(img)
dead_image = np.log10(img_freq)

cutoff = 120
img_freq_copy  = copy.deepcopy(img_freq)
img = np.fft.ifft2(img_freq)
plt.imshow(np.log10(np.abs(img)), cmap=plt.cm.gray)
plt.colorbar()
plt.savefig("ex2.pdf")

# ex 3
pixel_noise = 200

noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=np.shape(img))
img  = img_stash + noise 

cutoff = 200 
denoised_img = np.fft.fft2(img)

#acum vrem sa recuperam imaginea originala din cea cu noise 
