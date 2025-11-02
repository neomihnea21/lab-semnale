import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as sf
def create_specter(signal):
    step = len(signal)//200
    N = len(signal)
    index = 0
    frame = 0
    spectrogram = np.zeros(shape=(250, 2*step))
    while index + 2*step < N:
        chunk = signal[index: index+2*step]
        transform = np.fft.fft(chunk)
        spectrogram[frame] = 10*np.log10(np.abs(transform))
        index += step
        frame += 1
    plt.pcolormesh(spectrogram.T)
    plt.xticks()
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig("ex6.pdf")
_, data = sf.read("vocale.wav")
create_specter(data)
