import numpy as np
import matplotlib.pyplot as plt

base_time = np.linspace(-3, 3, 300)
times = []
times.append(np.linspace(-3, 3, 7))
times.append(np.linspace(-3, 3, 11))
times.append(np.linspace(-3, 3, 13))
times.append(np.linspace(-3, 3, 25))


B = 0.6

def rebuilt_sinc(valued_samples, samples, t, period):
    sinc_arg = (t - samples) / period
    return np.dot(valued_samples, np.sinc(sinc_arg)) 

fig, axs = plt.subplots(4)
for i in range(4):
    period =  6/(len(times[i])-1)
    points = times[i]
    valued_points=np.sinc(points)**2
    axs[i].plot(base_time, np.array([(np.sinc(B*t)**2) for t in base_time]), color='#e7fa00')
    axs[i].stem(times[i], np.array([(np.sinc(B*t)**2) for t  in times[i]]))
    axs[i].plot(base_time, np.array([rebuilt_sinc(valued_points, points, dt, period) for dt in base_time]), 'm--')
plt.savefig("ex1_B=0.6.pdf")


