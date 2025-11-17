import numpy as np
import matplotlib.pyplot as plt


points = np.random.uniform(-1, 1, 400)
fig, axs = plt.subplots(4)
axs[0].hist(points, bins=30)
for i in range(3):
    points = points * points
    axs[i+1].hist(points, bins=30)
plt.savefig("ex2.pdf")
