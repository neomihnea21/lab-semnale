import numpy as np
import matplotlib.pyplot as plt

base_time = np.linspace(-3, 3, 300)
times = []
times.append(np.linspace(-3, 3, 7))
times.append(np.linspace(-3, 3, 10))
times.append(np.linspace(-3, 3, 14))
times.append(np.linspace(-3, 3, 28))


B = 1

def rebuilt_sinc(times, period, t):
    ans = 0
    for i in range(len(times)):
      ans += np.sinc(times[i]*B)*np.sinc(t/period - len(times))
    return ans

fig, axs = plt.subplots(4)
for i in range(4):
    period  = 6 / (len(times[i])-1)
        
    axs[i].plot(base_time, np.array([np.sinc(B*t) for t in base_time]), color='#e7fa00')
    axs[i].stem(times[i], np.array([np.sinc(B*t) for t  in times[i]]))
    axs[i].plot(base_time, np.array([rebuilt_sinc(times[i], period, dt) for dt in base_time]), 'm--')
plt.savefig("ex1.pdf")


