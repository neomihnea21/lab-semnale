import numpy as np

#assume p and q are lists
def multiply_naive(p, q):
    # pad them a bit so the degrees match 
    deg_p = len(p) - 1
    deg_q = len(q) - 1
    if deg_p < deg_q:
        p = np.pad(p, deg_q-deg_p, constant_values=0)
    else:
        q = np.pad(q, deg_p-deg_q, constant_values=0)

    answer = np.zeros(deg_q+deg_p+1)
    for i in range (deg_p+1): 
        for j in range (deg_q+1):
           answer[i+j] += p[i]*q[j]
    return answer
def multiply_fft(p, q):
    limit = max(len(p)-1, len(q)-1)
    power = 1
    while power<limit:
        power *= 2
    power *= 4
    p=np.pad(p, (0, power-len(p)))
    q=np.pad(q, (0, power-len(q)))
    p_trans = np.fft.fft(p)
    q_trans = np.fft.fft(q)
    return np.real(np.fft.ifft(p_trans*q_trans))

p1 = np.array([1, 2])
p2 = np.array([1, 3])
print(multiply_fft(p1, p2))