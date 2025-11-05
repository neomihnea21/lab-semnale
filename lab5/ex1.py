import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
# a) - Frecventa de esantionare este de 1/h, sau 1/3600 Hz.
# b) - Esantioanele sunt sortate, primul este de pe 25-08-2012, al doilea de pe 25-09-2014, asa ca au fost obtinute in 25 de luni.

# c) - Stim ca semnalul original poate fi recoonstruit dintr-o esantionare de frecventa f = 1/3600 Hz
# => din teorema Shannon-Nyquist, el nu are componenete de frecventa >= 1 / 2f = 1800 Hz. 
# dar esantionarea este optima, asa ca frecventa maxima din semnal este de 1800 Hz.

# d)
data = pd.read_csv("Train.csv", delimiter=',')

data =  np.array(data["Count"]) # nu ne pasa de id-uri si timestampul de achizitie

N = len(data)
transform = np.fft.fft(data[:16384])
transform = np.abs(transform/N)


freqs = (1/3600) * np.linspace(0, 8192, 8192) / 16384 

plt.plot(freqs, transform[:8192])
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.savefig("d.pdf")

#e)
avg  = np.mean(data)
print(avg)
data = np.float64(data)
data -= np.mean(data)
# Putem calcula media ca fiind 138.95, deci exista o componenta continua, de valoare 138.95 (o sinusoida are media 0)

#f)
max_indices = np.argpartition(data, -4)[-4:]
for i in range(4):
    print(f"Componenta importanta {i} este la frecventa: "+ str(freqs[min(max_indices[i], 16384-max_indices[i])]) + "Hz")

#g)
# nota: o luna are 24*30 = 720 de esantioane
plt.clf()
data += avg
month_section =  data[1392:1392+720]
time = np.linspace(0, 720, 720)
plt.plot(time, month_section)
plt.savefig("g.pdf")

# h)
# In sinusioda, exista maxime pentru zile ale saptamanii si maxime anuale (de pilda, de Craciun este un spike
# si in fiecare vineri e un spike mai mic )
# asa ca am putea determina in ce zile este vineri, si in ce zi este Craciunul
# si de aici, am putea deduce, de exemplu, ca prima zi pentru care avem date este o luni, 26 august
# nu exista atat de multi ani recenti in care este 26 august e o luni.

# i) 
# nota: vom aplica filtrarea doar pe prima jumatate, a doua este la fel

plt.clf()
for i in range(0, 1024):
    transform[i] = 0
    transform[16383-i] = 0
time = np.linspace(0, 16384, 16384)
filtered_signal = np.fft.ifft(transform)
plt.plot(time, np.abs(filtered_signal))
plt.savefig("i.pdf")