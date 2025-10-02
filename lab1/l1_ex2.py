import numpy as np
import matplotlib.pyplot as plt

# a
times = np.linspace(0, 0.1, 40)
signal = np.array([np.cos(800*np.pi*t) for t in times])
plt.plot(times, signal)
plt.stem(times, signal)
plt.xlabel("Timp")
plt.ylabel("Amplitudine")
plt.savefig("samples.pdf")

# b
plt.clf()
times=np.linspace(0, 3, 800)
signal = np.array([np.cos(1600*np.pi*t) for t in times])
plt.plot(times, signal)
plt.xlabel("Timp")
plt.ylabel("Amplitudine")
plt.savefig("given_time.pdf")

# c
plt.clf()
times=np.linspace(0, 0.1, 192)
signal=np.array([240*t-np.floor(240*t) for t in times])
plt.plot(times, signal)
plt.xlabel("Timp")
plt.ylabel("Amplitudine")
plt.savefig("sawtooth.pdf")

# d
plt.clf()
times=np.linspace(0, 0.1, 150)
signal=np.array([np.sign(np.sin(600*np.pi*t)) for t in times])
plt.plot(times, signal)
plt.xlabel("Timp")
plt.ylabel("Amplitudine")
plt.savefig("square.pdf")

#e
plt.clf()
signal=np.random.rand(128, 128)
plt.imshow(signal, origin='lower', cmap='plasma')
plt.xlabel("Timp")
plt.ylabel("Amplitudine")
plt.savefig("random.pdf")

#f
plt.clf()
signal=np.zeros((128, 128))
for i in range(128):
    for j in range(128):
        if (i+j)%2==0:
            signal[i][j]=1
plt.imshow(signal, origin='lower', cmap='plasma')
plt.xlabel("Timp")
plt.ylabel("Amplitudine")
plt.savefig("checkerboard.pdf")