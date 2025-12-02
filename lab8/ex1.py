import numpy as np
import matplotlib.pyplot as plt
import copy
N = 1000
time = np.linspace(0, 8, N)

trend = np.array([5*x*x-2*x+4 for x in time])
season = np.array([10*np.sin(2*np.pi*x) + 12*np.sin(4*np.pi*x + np.pi/3) for x in time])
noise = np.random.normal(0, 5, N)

plt.subplot(4, 1, 1)
plt.plot(time, trend+season+noise)

plt.subplot(4, 1, 2)
plt.plot(time, trend)

plt.subplot(4, 1, 3)
plt.plot(time, season)

plt.subplot(4, 1, 4)
plt.plot(time, noise)

plt.savefig("1a.pdf")
plt.clf()

# b)
# conform https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
y = trend+season+noise
def numpy_autocorr(series):
    result = np.correlate(series, series, 'full')
    return result[len(result)//2:]

def my_autocorr(series, p):
    Y = np.array(series[i:i+p] for i in range(N-p))
    #ne va trebui un semnal de dimensiune m cand aplicam Yule-Walker
    #asa ca ne taiem o bucata potrivita
    snippet=copy.deepcopy(series[p:N])
    snippet = np.reshape(snippet, (-1, ))
    Gamma = Y.T @ Y
    gamma = Y.T @ snippet
    return gamma
    
ans = numpy_autocorr(y)
plt.plot(ans)
plt.savefig("1b.pdf")
plt.clf()


# c)
# vom folosi m=990 and p=10 
y_hat = np.zeros(N)
p = 10
for i in range(p):
    y_hat[i] = y[i]
for i in range(p+1, N):
    for j in range(i, i-p, -1):
        y_hat[i] += (1/p) * y[j]

plt.plot(time, y_hat, label='Predicted sales')
plt.plot(time, y, label = 'Actual sales')
plt.xlabel("Time")
plt.ylabel('Sales')
plt.legend()
plt.savefig("1c.pdf")

# d)

def predict(series, m, p):
    if (m+p < N):
        raise ValueError("Nu putem prezice")
    Y = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            Y[i][j]=series[(i+j)%N]
    #ne va trebui un semnal de dimensiune m cand aplicam Yule-Walker
    #asa ca ne taiem o bucata potrivita
    snippet=np.take(series, np.arange(p, p+m), mode='wrap')
    snippet = np.reshape(snippet, (-1, 1))
    Gamma = Y.T @ Y
    gamma = np.matmul(Y.T, snippet)
    ans = np.matmul(np.linalg.inv(Gamma), gamma)
    err = 0
    for i in range(m, m+p):
        err += (ans[i-m][0] - series [i%N]) ** 2 
    return ans, err/N


min_err = 1e12 # sau altceva foarte mare
min_p = 0
min_m = 0
# bucata asta e nebun de alb, captureaza CPU-ul pentru vesnicie
# probabil merge un soi de gradient descent? functia pare derivabila
for p in range(N):
    for m in range(N, N-p, -1):
        y_hat, err = predict(y, m, p)
        if err < min_err:
            min_err = err
            min_p=p
            min_m=m
print(f"Eroarea minima se obtine pentru p={min_p} si m={min_m}")