import numpy as np
import matplotlib.pyplot as plt

times = np.linspace(0, 1, 400)
sine = np.array([np.sin(4*np.pi*t) for t in times])
square = np.sign(sine)

fig, axs = plt.subplots(3)
axs[0].plot(times, sine)
axs[1].plot(times, square)
axs[2].plot(times, sine+square)

plt.savefig("ex4.pdf")
