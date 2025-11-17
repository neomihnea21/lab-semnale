import numpy as np
import matplotlib.pyplot as plt
shift=3

x = np.random.uniform(-1, 1, 20)
y = np.roll(x, shift=shift)

recovered_correlations = np.fft.ifft(np.fft.fft(x) * np.fft.fft(y))

recovered_correlations_2 = np.fft.ifft(np.fft.fft(x) / np.fft.fft(y))

plt.stem(np.real(recovered_correlations_2))
plt.savefig("ex4.pdf")
print(np.argmax(np.abs(recovered_correlations)))
