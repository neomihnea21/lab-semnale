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
Y = np.zeros((m, p))
for i in range(m):
    for j in range(p):
        Y[i, j] = y[N-i-j-1]

def MSE(a, b):
    return np.mean((a-b)**2)
# aici in biblioteca nu exista lambda
def l1_reg(y, Y):
   return reg.l1regls(Y, y)

def greedy_reg(y, Y, regs=4):

    features = np.shape(Y)[0]
    curr_regressors = []
    test_regressors = []
    best_error = 1e9
    for _ in range(regs):
       best_i = -1
       for i in range(features):
           if i not in curr_regressors:
               test_regressors = curr_regressors + [i]
               atoms = Y[test_regressors, :] # the term is borrowed from dictionary learning, since it's he same idea as OMP
               coefs, _, _, _ = np.linalg.lstsq(atoms, y)
               pred = atoms @ coefs
               error = MSE(pred, y)
               if error < best_error:
                   best_error = error
                   best_i = i
       # trick: if 2 regressors do better than 3, we don't bother picking a 3rd
       if best_i != -1:
          curr_regressors += [best_i]       
           # evaluate the current set of regressors
    return np.array(curr_regressors)

#5
def check_stationary(params):
    roots = get_roots(params)
    is_stationary = True
    for root in roots:
        if np.abs(root) < 1:
            is_stationary = False
    return is_stationary

print(check_stationary(x))