import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0, 1, 1000)

signal = np.array([np.sin(12*np.pi*t) for t in time])

flower1 = np.array([signal[i] * (np.e **(2*np.pi*1j*time[i])) for i in range(1000)])
flower2 = np.array([signal[i] * (np.e **(4*np.pi*1j*time[i])) for i in range(1000)])
flower3 = np.array([signal[i] * (np.e **(6*np.pi*1j*time[i])) for i in range(1000)])
flower4 = np.array([signal[i] * (np.e **(12*np.pi*1j*time[i])) for i in range(1000)])

real = np.array([x.real for x in flower1])
imag = np.array([x.imag for x in flower1])

fig, axs  = plt.subplots(2, 2)
axs[0][0].scatter(flower1.real, flower1.imag, c=np.abs(flower1), cmap='inferno')
axs[1][0].scatter(flower2.real, flower2.imag, c=np.abs(flower2), cmap='inferno')
axs[0][1].scatter(flower3.real, flower3.imag, c=np.abs(flower3), cmap='inferno')
axs[1][1].scatter(flower4.real, flower4.imag, c=np.abs(flower4), cmap='inferno')

plt.savefig("ex2.pdf")