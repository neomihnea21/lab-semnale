import numpy as np
import scipy.io.wavfile as wav
import sounddevice 

times = np.linspace(0, 1, 44100)
signal1 = np.array([np.sin(2000*np.pi*t) for t in times])
signal2 = np.array([np.sin(3000*np.pi*t) for t in times])

signal = np.concat((signal1, signal2))
wav.write("ex5.wav", 44100, signal)

