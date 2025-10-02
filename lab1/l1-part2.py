import numpy as np
import matplotlib.pyplot as plt

reduced_times=np.linspace(0, 0.03, 6)
full_times=np.linspace(0, 0.03, 54)
x_signal = np.array([np.cos(520*np.pi*t+np.pi/3) for t in reduced_times])
y_signal = np.array([np.cos(280*np.pi*t-np.pi/3) for t in reduced_times])
z_signal = np.array([np.cos(120*np.pi*t+np.pi/3) for t in reduced_times])

x_signal_extended = np.array([np.cos(520*np.pi*t+np.pi/3) for t in full_times])
y_signal_extended = np.array([np.cos(280*np.pi*t-np.pi/3) for t in full_times])
z_signal_extended = np.array([np.cos(120*np.pi*t+np.pi/3) for t in full_times])
fig, axs = plt.subplots(3)
fig.suptitle("Esantionari")
axs[0].plot(full_times, x_signal_extended)
axs[1].plot(full_times, y_signal_extended)
axs[2].plot(full_times, z_signal_extended)
axs[0].stem(reduced_times, x_signal)
axs[1].stem(reduced_times, y_signal)
axs[2].stem(reduced_times, z_signal)

plt.savefig("sampled_plots.pdf")