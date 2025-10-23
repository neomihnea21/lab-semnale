import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0, 1, 16)

part1 = np.array([np.sin(6*np.pi*t) for t in time] )
part2 = np.array([np.sin(8*np.pi*t + np.pi/3) for t in time])
part3 = np.array([np.sin(10*np.pi*t - np.pi/3)  for t in time])

signal = part1 + part2 + part3

fourier = np.zeros(dtype=np.complex128, shape=(16, 16))
for i in range(16):
    for j in range(16):
        fourier[i][j] = np.e ** (2* np.pi * 1j * i *j /16)


amplitudes = np.matmul(signal, fourier)
amplitudes = np.abs(amplitudes)

freqs = np.array([i for i in range(1, 17)])
plt.stem(freqs, amplitudes)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.savefig("ex3.pdf")