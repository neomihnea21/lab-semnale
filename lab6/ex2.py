import numpy as np
import matplotlib.pyplot as plt


points = np.random.uniform(-1, 1, 400)
fig, axs = plt.subplots(4)
axs[0].hist(points, bins=30)
for i in range(3):
    points = np.convolve(points, points)
    axs[i+1].hist(points, bins=30)

#Observam ca punctele converg spre o distributie Gaussiana N(0, 1). Aceasta este o consecinta a celebrei Teoreme Limita Centrala.
plt.savefig("ex2.pdf")
