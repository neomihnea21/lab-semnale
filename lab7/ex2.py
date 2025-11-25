import numpy as np
import matplotlib.pyplot as plt
import scipy.datasets as data
import copy

def SNR(image):
   return np.log10(np.mean(image)/np.std(image))

#functia presupune ca imaginea e deja in frecventa
def denoise(image, cutoff):
   image_decibels = 20*np.log10(np.abs(img))
   image_cutoff = copy.deepcopy(image)
   image_cutoff[image_decibels > cutoff] = 0
   return image_cutoff

img = data.face(gray=True)
img_stash = copy.deepcopy(img)

img_freq = np.fft.fft2(img)

cutoff = 120
img_freq=denoise(img_freq, cutoff=cutoff)

img = np.fft.ifft2(img_freq)
plt.imshow(np.real(img), cmap=plt.cm.gray)
plt.colorbar()
plt.savefig("ex2.pdf")

# ex 3
pixel_noise = 200

noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=np.shape(img))
img  = img_stash + noise 
print(f"SNR in decibeli: {SNR(img)} pe blurat, fata de {SNR(img_stash)} pe initial")
plt.imshow(img, cmap=plt.cm.gray)
plt.savefig("ex3-noised.pdf")

plt.clf()
# now cut the noise
img_freq = np.fft.fft2(img)
img_cutoff = denoise(img_freq, 200)
img_cutoff  = np.fft.ifft2(img_cutoff)
plt.imshow(np.real(img_cutoff), cmap=plt.cm.gray)
print(f"SNR dupa zgomot pus si scos: {SNR(np.real(img_cutoff))}")
plt.savefig("ex3-noised-and-denoised.pdf")
