import numpy as np
import scipy.io.wavfile as wav
import sounddevice
times = np.linspace(0, 20, 132300)
signal = 300 * np.array([np.sin(1000*np.pi*t) for t in times])

wav.write("ex3.wav", 44100, signal)
sounddevice.play(signal, 44100)

data, rate = wav.read("ex3.wav")