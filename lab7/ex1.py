import numpy as np
import matplotlib.pyplot as plt

def time_1(n1, n2):
    return np.sin(2*np.pi*n1+3*np.pi*n2)

def time_2(n1, n2):
    return np.sin(4*np.pi*n1)+np.cos(6*np.pi*n2)

def freq_1(n1, n2, N):
    if n1 ==0 and (n2 == 5 or n2 == N-5):
        return 1
    return 0
def freq_2(n1, n2, N):
    if n2 == 0 and (n1==5 or n1 == N-5):
        return 1
    return 0
def freq_3(n1, n2, N):
    if n1 == n2 and (n1 == 5 or n1 == N-5):
        return 1
    return 0

times = [time_1, time_2]
freqs = [freq_1, freq_2, freq_3]
img = np.zeros((600, 600))
ct=1
for time in times:
  for i in range(600):
     for j in range(600):
         img[i][j]=time(i, j)
  plt.imshow(np.log10(img), cmap='Greys')
  plt.colorbar()
  plt.savefig(f"time_{ct}.pdf")
  plt.clf()
  ct+=1

ct=1
for freq in freqs:
    for i in range(600):
        for j in range(600):
            img[i][j]=freq(i, j, 600)
    img = np.fft.ifft2(img)
    plt.imshow(np.log10(np.real(img)), cmap='Greys')
    plt.colorbar()
    plt.savefig(f"Frequency_{ct}.pdf")
    plt.clf()
    ct+=1