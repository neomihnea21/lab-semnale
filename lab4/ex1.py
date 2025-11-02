import numpy as np
import time
import matplotlib.pyplot as plt
def get_fft(sample):
   if len(sample) == 1:
      return sample
   odd = get_fft(sample[1::2])
   even = get_fft(sample[::2])
    
   n = len(sample)
   answer = np.zeros(n, dtype=np.complex128)
    
   for k in range (len(sample) // 2):
       half_transform = np.exp(-1j*2*np.pi*k) * odd
       answer[k] = even[k] + half_transform[k]
       answer[k+n//2] = even[k] - half_transform[k]
   return answer
dft_times = []
fft_times = []
numpy_times = []
size = 128
while size < 16384:
    coords = np.arange(size)
    basis = np.outer(coords, coords)
    exponents = 2 * np.pi * 1j * basis / size
    fourier = np.exp(exponents)
    sample = np.random.rand(size)
    t0 = time.time()
    ans_dft  = np.matmul(fourier, sample)
    t1=time.time()
    dft_times.append(t1-t0)

    t0 = time.time()
    ans_numpy = np.fft.fft(sample)
    t1=time.time()
    numpy_times.append(t1-t0)

    t0 = time.time()
    get_fft(sample)
    t1 = time.time()
    fft_times.append(t1-t0)

    size *=2
    
xticks = np.arange(len(fft_times))
plt.plot(xticks, dft_times, color = '#290bb3', label = 'DFT')
plt.plot(xticks, fft_times, color = '#d930e6', label = 'Handmade FFT')
plt.plot(xticks, numpy_times, color = '#8bed4e', label = 'Numpy FFT')
plt.legend()
plt.yscale('log')
plt.savefig("ex1.pdf")
