import numpy as np
import matplotlib.pyplot as plt

fourier = np.zeros(dtype=np.complex128, shape=(8, 8))
for i in range(8):
    for j in range(8):
        fourier[i][j] = np.e ** (2* np.pi * 1j * i *j /8)



fig, axs = plt.subplots(8)
time = np.linspace(0, 1, 8)
for i in range(len(fourier)):
    real = np.array([x.real for x in fourier[i]])
    imag = np.array([x.imag for x in fourier[i]])
    axs[i].plot(time, real, color='#87CEEB')
    axs[i].plot(time, imag, '--', color='#50C878')
plt.savefig("waves.pdf")

# now, let's check our work
fourier_hermitian = np.conjugate(fourier.T)

test = np.matmul(fourier, fourier_hermitian)
print(np.allclose(test, 8*np.identity(8)))