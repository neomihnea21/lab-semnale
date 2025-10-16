import numpy as np
import matplotlib.pyplot as plt

times = np.linspace(-np.pi/2, np.pi/2, 600)
def sin_taylor(x):
    return x - (x**3)/3 + (x**5)/5

def sin_pade(x):
    return (x-7*(x**3)/60)/(1+(x**2)/20)

linear = np.array([t for t in times])
sine = np.array([np.sin(t) for t in times])
sine1 = np.array([sin_taylor(t) for t in times])
sine2 = np.array([sin_pade(t) for t in times])
fig, axs = plt.subplots(2)
axs[0].plot(times, sine)
axs[0].plot(times, linear)
axs[1].plot(times, linear-sine)
axs[1].set_yscale('log')
plt.savefig("ex8-part1.pdf")

plt.clf()
plt.plot(times, sine2-linear)
plt.yscale('log')
plt.savefig("ex8-part2.pdf")
