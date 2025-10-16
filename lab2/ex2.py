import numpy as np
import matplotlib.pyplot as plt

times = np.linspace(0, 1, 400)
signal1 = 3 * np.array([np.sin(5*np.pi*t) for t in times])
signal2 = 3 * np.array([np.sin(5*np.pi*t + np.pi/2) for t in times])
signal3 = 3 * np.array([np.sin(5*np.pi*t + np.pi) for t in times])
signal4 = 3 * np.array([np.sin(5*np.pi*t - np.pi/2) for t in times])

plt.plot(times, signal1)
plt.plot(times, signal2)
plt.plot(times, signal3)
plt.plot(times, signal4)
plt.savefig("ex2-part1.pdf")
