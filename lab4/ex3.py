import numpy as np
import matplotlib.pyplot as plt

fake_time = np.linspace(0, 1, 5)
true_time = np.linspace(0, 1, 1024)
signal1 = np.array([np.sin(2*np.pi*t) for t in true_time])
signal2 = np.array([np.sin(10*np.pi*t) for t in true_time])
signal3 = np.array([np.sin(0.9*np.pi*t) for t in true_time])

points = np.array([np.sin(2*np.pi*t) for t in fake_time])

fig, axs = plt.subplots(3)
axs[0].stem(fake_time, points)
axs[0].plot(true_time, signal1, color='#0233CC')

axs[1].stem(fake_time, points)
axs[1].plot(true_time, signal2, color='#79DB2E')

axs[2].stem(fake_time, points)
axs[2].plot(true_time, signal3, color='r')

plt.savefig("ex3.pdf")