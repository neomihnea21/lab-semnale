import numpy as np

num_samples = 600

def get_hann(n):
    return 0.5*(1-np.cos(2*np.pi*n/num_samples))

