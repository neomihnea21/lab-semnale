import numpy as np
import matplotlib.pyplot as plt


points = np.random.uniform(-1, 1, 400)
fig, axs = plt.subplots(4)
axs[0].hist(points, bins=30)
axs[0].set_xlabel("Initial")
for i in range(3):
    points = np.convolve(points, points)
    axs[i+1].hist(points, bins=30)
    axs[i+1].set_xlabel(f"Dupa {i+1} convolutii")

#Observam ca punctele converg spre o distributie Gaussiana N(0, 1). Aceasta este o consecinta a celebrei Teoreme Limita Centrala.
plt.savefig("ex2.pdf")
