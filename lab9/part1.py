import numpy as np 

import matplotlib.pyplot as plt

# 1
N = 1000
time = np.linspace(0, 8, N)

trend = np.array([5*x*x-2*x+4 for x in time])
season = np.array([10*np.sin(2*np.pi*x) + 12*np.sin(4*np.pi*x + np.pi/3) for x in time])
noise = np.random.normal(0, 5, N)

y = trend + season + noise

#2
def exponential(alpha):
    s = np.zeros(N)
    s[0]=y[0]

    for t in range(1, N):
        s[t] = (1-alpha)**t
        for j in range(t):
            s[t] += alpha*y[t-j]*(1-alpha)**j
    return s

def double_exponential(alpha, beta, m):
    s = np.zeros(N)
    b = np.zeros(N)
    y_hat = np.zeros(N)
    s[0] = y[0]
    b[0] = y[1] - y[0]
    
    for t in range(1, N-1):
        s[t] = alpha*y[t] + (1-alpha) * (s[t-1] - b[t-1])
        b[t] = beta*(s[t]-s[t-1]) + (1-beta)*b[t-1]
    for i in range(N):
        if i<m:
            y_hat[i] = 0
        else:
            y_hat[i] = s[i-m] + m*b[i-m]
    return y_hat
def triple_exponential(alpha, beta, gamma, m, L):
    s = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(L)
    y_hat = np.zeros(N)

    s[0] = y[0]
    b[0] = y[1] - y[0]
    c[0] = 0
    for t in range(1, N-1):
        s[t] = alpha*y[t] + (1-alpha) * (s[t-1] - b[t-1])
        b[t] = beta*(s[t]-s[t-1]) + (1-beta)*b[t-1]
        if t<L:
            c[t] = 0
        else:
            c[t%L] = gamma * y[t]/s[t] + (1-gamma) * c[t%L] 
    for i in range(N):
        if i<m:
          y_hat[i]=0
        else:
          y_hat[i]=(s[i-m] + m*b[i-m])*c[(t+m)%L]
    return y_hat


def series_error(s, x):
    err = 0
    for i in range(N-1):
       err += (s[i]-x[i+1]) ** 2
    return err/N


def optimize_2D(iter):
    best_alpha=0.5
    best_beta=0.5
    best_error = 1e12
    alpha = 0.5
    beta = 0.5
    step = 0.1
    for _ in range(iter):
       alpha = best_alpha
       beta = best_beta
       for i in range(-5, 6):
           for j in range(-5, 6):
               series = double_exponential(alpha+i*step, beta+j*step, 5)
               if series_error(series, y) < best_error:
                   best_alpha = alpha + i*step
                   best_beta = beta + j*step
                   best_error = series_error(series, y)
       step /= 10
    return best_alpha, best_beta, best_error
            
def optimize_3D(iter):
    best_alpha=0.5
    best_beta=0.5
    best_gamma = 0.5
    best_error = 1e12
    alpha = 0.5
    beta = 0.5
    gamma = 0.5
    step = 0.2
    for _ in range(iter):
       alpha = best_alpha
       beta = best_beta
       gamma = best_gamma
       for i in range(-2, 3):
           for j in range(-2, 3):
               for k in range(-2, 3):
                series = triple_exponential(alpha+i*step, beta+j*step, gamma+k*step, 5, 10)
                if series_error(series, y) < best_error:
                    best_alpha = alpha + i*step
                    best_beta = beta + j*step
                    best_gamma = gamma + k*step 
                    best_error = series_error(series, y)
       step /= 10
    return best_alpha, best_beta, best_gamma, best_error

best_alpha=0
best_error=1e29
for i in range(100):
    alpha = i/100
    err = series_error(exponential(alpha), y)
    plt.plot(alpha, err)
    if err<best_error:
        best_error = err
        best_alpha = alpha
print(f"Cel mai bun alpha pentru exponentiala simpla este {best_alpha}, si eroarea este {best_error}")
plt.savefig("simple_exp.pdf")
plt.clf()

alpha, beta, err = optimize_2D(2)
print(f"Cea mai buna estimare 2D este pentru alpha={alpha} si beta={beta}, eroarea este {err}")

alpha, beta, gamma, err = optimize_3D(3)
print(f"Cea mai buna estimare 3D este pentru alpha={alpha}, beta={beta} si gamma={gamma} eroarea este {err}")
# pe cazurile 2D si 3D nu putem face plot, pentru ca sunt prea multe valori de calculat