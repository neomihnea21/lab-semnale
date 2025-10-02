import numpy as np
import matplotlib.pyplot as plt

times=np.linspace(0, 0.03, 60)

x_signal = np.array([np.cos(520*np.pi*t+np.pi/3) for t in times])
y_signal = np.array([np.cos(280*np.pi*t-np.pi/3) for t in times])
z_signal = np.array([np.cos(120*np.pi*t+np.pi/3) for t in times])

fig, axs = plt.subplots(3)
fig.suptitle("Sinusoide simple")
axs[0].plot(times, x_signal)
axs[1].plot(times, y_signal)
axs[2].plot(times, z_signal)

plt.savefig("basic_plots.pdf")
