import numpy as np
import statsmodels.tsa.arima.model
import l1regls as reg
#4 
def get_roots(coefs):
    N = len(coefs)
    companion = np.zeros((N, N))
    for i in range(N-1):
        companion[i+1][i] = 1
    for i in range(N):
        companion[i][N-1] = -coefs[i]
    
    eigens, _ = np.linalg.eig(companion)
    return eigens

#1
N = 1000
time = np.linspace(0, 8, N)

trend = np.array([5*x*x-2*x+4 for x in time])
season = np.array([10*np.sin(2*np.pi*x) + 12*np.sin(4*np.pi*x + np.pi/3) for x in time])
noise = np.random.normal(0, 5, N)

y = trend + season + noise

#2 
#Vom lua coeficientii pentru AR din STATSMODELS
p = 25
m = 25
model = statsmodels.tsa.arima.model.ARIMA(y, order=(p, 0, 0))
results = model.fit()

# 3 
# setup pentru regularizarea cu OMP / L1
x = results.params[1:]
Y = np.zeros(m, p)
for i in range(m):
    for j in range(p):
        Y[i, j] = y[N-i-j]

def l0_reg(y, Y):
   return reg.l1regls(Y, x)
#5
def check_stationary(params):
    roots = get_roots(params)
    is_stationary = True
    for root in roots:
        if np.abs(root) < 1:
            is_stationary = False
    return is_stationary
print(check_stationary(x))