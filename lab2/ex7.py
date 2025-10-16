import numpy as np
import matplotlib.pyplot as plt
times = np.linspace(0, 1, 1000)

signal1 = np.array([np.sin(6*np.pi*t) for t in times])
signal2 = np.array([np.sin(6*np.pi*t) for t in times[0::4]])
signal3 = np.array([np.sin(6*np.pi*t) for t in times[1::4]])

fig, axs = plt.subplots(3)
axs[0].plot(times, signal1)
axs[1].plot(times[0::4], signal2)
axs[2].plot(times[1::4], signal3)
plt.savefig("ex7.pdf")

#efectul: desi este greu de vazut, sinusoida are "rezolutie mai mica": s-ar vedea  mai clar daca am pastra doar fiecare al 25-lea element