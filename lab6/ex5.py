import numpy as np
import matplotlib.pyplot as plt
num_samples = 600

def get_hann(n, omega):
    if n>=0 and n < omega:  
      return 0.5*(1-np.cos(2*np.pi*n/num_samples))
    else:
      return 0

def get_window(n, omega):
   if n in range(0, omega):
      return 1
   else:
      return 0

hanning_filter = np.array([get_hann(n, 200) for n in range(200)])
rect_filter = np.array([get_window(n, 200) for n in range(200)])

f = 100
time = np.linspace(0, 0.5, 300)
signal = np.array([np.sin(2*np.pi*f*t) for t in time])

pass_1 = np.convolve(signal, hanning_filter)
pass_2 = np.convolve(signal, rect_filter)
fig, axs = plt.subplots(2)
plt.suptitle("Ferestre Hanning(sus) si dreptunghiulara(jos)")
axs[0].plot(np.arange(len(pass_1)), pass_1)
fig.supxlabel("Time")
fig.supylabel("Amplitude")
axs[1].plot(np.arange(len(pass_2)), pass_2)
plt.savefig("ex5.pdf")