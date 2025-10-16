import numpy as np
import matplotlib.pyplot as plt

times = np.linspace(0, 1, 400)
signal = 3 * np.array([np.sin(5*np.pi*t) for t in times])
noise = np.random.normal(0, 1, 400)

ratios = [0.1, 1, 10, 100]
fig, axs = plt.subplots(4)

for i in range (len(ratios)):
    gamma = np.linalg.norm(signal)/(np.linalg.norm(noise)*(np.sqrt(ratios[i])))
    new_signal = signal + gamma * noise
    axs[i].plot(times, new_signal)
plt.savefig("ex2-part2.pdf")
