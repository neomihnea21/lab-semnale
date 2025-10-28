import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0, 1, 512)

part1 = np.array([np.sin(6*np.pi*t) for t in time] )
part2 = np.array([np.sin(8*np.pi*t + np.pi/3) for t in time])
part3 = np.array([np.sin(10*np.pi*t - np.pi/3)  for t in time])

signal = part1 + part2 + part3

fourier = np.zeros(dtype=np.complex128, shape=(512, 512))
for i in range(512):
    for j in range(512):
        fourier[i][j] = np.e ** (2* np.pi * 1j * i *j /512)


amplitudes = np.matmul(signal, fourier)
amplitudes = np.abs(amplitudes)

freqs = np.array([i for i in range(1, 513)])

fig, axs = plt.subplots(2)

axs[0].plot(time, signal)
axs[1].stem(freqs, amplitudes)
plt.savefig("ex3.pdf")