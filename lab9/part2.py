import numpy as np 
import statsmodels
import matplotlib.pyplot as plt
import statsmodels.tsa
import statsmodels.tsa.arima
import statsmodels.tsa.arima.model
import scipy.optimize as optims
# 1
N = 1000
time = np.linspace(0, 8, N)

trend = np.array([5*x*x-2*x+4 for x in time])
season = np.array([10*np.sin(2*np.pi*x) + 12*np.sin(4*np.pi*x + np.pi/3) for x in time])
noise = np.random.normal(0, 5, N)

y = trend + season + noise

# 3
def get_moving_average(coefs, y, q):
    y_hat=np.zeros(N)
    for i in range(N):
        if i<q:
            y_hat[i] = y[i] 
        else:
            mu = np.mean(y[i-q:i])
            epsilon = y[i-q:i] - mu

            # trebuie sa gasim noii coeficienti, si rezolvam o problema least-squares
            # asa ca, pe moment, vom veni cu ei din afara
            y_hat[i] = np.dot(coefs, epsilon) + mu
    return y_hat


Q = 4

plt.plot(y, label='Actual')
plt.plot(get_moving_average(np.ones(Q), y, Q), label='Predicted')
plt.legend()
plt.savefig("3.pdf")
#4

plt.clf()
best_p = 0
best_q = 0
best_error = 1e12
MAX_HORIZON = 5

# nu putem testa 400 de combinatii de P si Q, dureaza prea mult. In acest exemplu, vom testa 25, dar se extinde 
for p in range(MAX_HORIZON):
    for q in range(MAX_HORIZON):
        model = statsmodels.tsa.arima.model.ARIMA(y, order=(p, 0, q))
        fitted_model = model.fit()
        error = fitted_model.aic
        if error<best_error:
            best_error = error
            best_p = p
            best_q = q
        
model = statsmodels.tsa.arima.model.ARIMA(y, order=(best_p, 0, best_q)).fit()
fitted_model = model.predict()

plt.title(f"ARMA({best_p}, {best_q})")
plt.plot(time, y, label = 'Actual')
plt.plot(time, fitted_model, label = 'Predicted')
plt.legend()
plt.savefig("4.pdf")