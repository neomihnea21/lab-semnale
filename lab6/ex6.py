import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy
import scipy.signal
# a)
df = pd.read_csv("Train.csv")
signal = df["Count"][:72]
# b)
W = 15
filtered_signal = np.convolve(signal, np.ones(W), 'valid') / W
plt.plot(filtered_signal)
plt.xlabel("Timp")
plt.ylabel("Amplitudine")
plt.savefig("simple_conv.pdf")

# c)
# Semnalul este esantionat cu frecventa de 1/3600 Hz (1/ora)
# De aici, nu putem determina "strict" frecventa Nyquist, dar, presupunand ca semnalul este esantionat corect,
# ne putem da seama ca nu exista frecvente mai mari de 1/7200 Hz. Vom considera valoarea asta pentru Nyquist.
# Vom filtra toate componentele cu frecventa mai mare de 1/21600 Hz. (deoarece sunt variabilitate restransa in cadrul unei saptamani)
# In atari conditii, frecventa relativa este 1/3.

# d)
plt.clf()
cutoff = 1/12000
butter = scipy.signal.butter(5, cutoff, 'low', fs=1/3600)

filtered_signal_butter = scipy.signal.filtfilt(*butter, signal)

cebisev = scipy.signal.cheby1(5, 5, cutoff, 'lowpass', fs=1/3600)
filtered_signal_cebisev = scipy.signal.filtfilt(*cebisev, signal)

#e)

fig, axs = plt.subplots(4)
axs[0].plot(np.arange(len(filtered_signal_butter)), filtered_signal_butter)
axs[2].plot(np.arange(len(filtered_signal_cebisev)), filtered_signal_cebisev)
axs[1].plot(signal)
axs[1].set_title("Semnal original")
axs[3].plot(signal)
axs[3].set_title("Semnal original")
plt.ylabel("Amplitudini")
plt.xlabel("Timpi")
plt.savefig("ex6-butter.pdf")
#As alege Butterworth, pentru ca este mai aproape de un trece-jos pur.

#f)
plt.clf()
orders = [4, 6, 8]
#coeficientii de atenuare se mai noteaza cu zeta
zetas = [3, 5]

fig, axs = plt.subplots(9)
for i in range(3):
    butter = scipy.signal.butter(orders[i], cutoff, 'low', fs=1/3600)
    butter_pass = scipy.signal.filtfilt(*butter, signal)
    axs[i].plot(np.arange(len(butter_pass)), butter_pass)
for i in range(3):
    for j in range(2):
        cheby = scipy.signal.cheby1(orders[i], zetas[j], cutoff, 'lowpass', fs=1/3600)
        filtered_cheby = scipy.signal.filtfilt(*cheby, signal)
        axs[3*j+i+3].plot(np.arange(len(filtered_cheby)), filtered_cheby)
        axs[3*j+i+3].set_xlabel("Timp")
        axs[3*j+i+3].set_ylabel("Semnal filtrat")
plt.savefig("ex6-many-filters.pdf")