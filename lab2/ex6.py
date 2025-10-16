import numpy as np
import matplotlib.pyplot as plt

basic_freq = 800
times = np.linspace(0, 1, basic_freq)
signal1=np.array([np.sin(basic_freq*np.pi*t) for t in times])
signal2=np.array([np.sin(basic_freq*np.pi*t/2) for t in times])
signal3=np.array([np.sin(0*t) for t in times])

fig, axs  = plt.subplots(3)
axs[0].plot(times, signal1)
axs[1].plot(times, signal2)
axs[2].plot(times, signal3)
plt.savefig("ex6.pdf")
