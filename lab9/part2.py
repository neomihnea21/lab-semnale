import numpy as np 
import statsmodels
import matplotlib.pyplot as plt
import statsmodels.tsa
import statsmodels.tsa.arima
import statsmodels.tsa.arima.model

# 1
N = 1000
time = np.linspace(0, 8, N)

trend = np.array([5*x*x-2*x+4 for x in time])
season = np.array([10*np.sin(2*np.pi*x) + 12*np.sin(4*np.pi*x + np.pi/3) for x in time])
noise = np.random.normal(0, 5, N)

y = trend + season + noise

# 3
def get_moving_average(y, q):
    y_hat=np.zeros(N)
    coefs = np.ones(q)
    sum = 0
    for i in range(N):
        if i<q:
            sum += y[i]
            y_hat[i] = y[i] 
        else:
            mu = sum / N
            epsilon = y_hat[i-q:i] - mu 
            
            sum += (y[i]-y[i-q])

#4 

p=12 
q=3 

model = statsmodels.tsa.arima.model.ARIMA(y, order=(p, 0, q)).fit()
print(model.summary())