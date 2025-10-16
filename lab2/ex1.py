import numpy as np
import matplotlib.pyplot as plt

times = np.linspace(0, 1, 400)
A = 2
sin_signal = A * np.array([np.sin(6*np.pi*t) for t in times])
cos_signal = A * np.array([np.cos(np.pi/2 - 6*np.pi*t) for t in times])

fig, (ax1, ax2) = plt.subplots(2)

ax1.plot(times, sin_signal)
ax2.plot(times, cos_signal)

plt.savefig("ex1.pdf")