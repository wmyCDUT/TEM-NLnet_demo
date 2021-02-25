import torch
import numpy as np
import matplotlib.pyplot as plt
WINDOW = 5
def smooth(a,WSZ):
    if a.shape[0] == 1:
            a = np.reshape(a, [400,])
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))


def g_noise(noise_level):
    
    sigma = noise_level
    p = np.random.randn(1, 400) * sigma
    return p

def is_preprocess(input):
          #print(input.shape)
          output = input - smooth(input,WINDOW)
          return output
def  is_preprocess_batch(input,n):
          out = np.empty([n, 1, 400])
          for i in range(n):
                 y = is_preprocess(input[i,0,:])
                 out[i, 0] = y
          return out

def simulation_data(sigma,noise_level,preprocess):
    t = np.linspace(0, 4, 400)
    k1 = np.random.randint(50000, 120000)
    #k1 = np.random.randint(400, 500)
    k2 = np.random.randint(10,40)
    b = np.random.randint(1450, 1550)
    y = k1 * np.exp(-k2 * t)  + b
    if sigma is True:
        noise = g_noise(noise_level)
        out = y + noise
        if preprocess:
                 out = is_preprocess(out)
    else:
        out = y
    return out


def batch_simulation_data(n, label_n,noise_level,is_preprocess):
    out = np.empty([n, 1, 400])
    for i in range(n):
        y = simulation_data(label_n,noise_level,is_preprocess)
        out[i, 0] = y
    
    return out
def test_simulation_data(n, noise_level):
    clean = np.empty([n, 1, 400])
    noisy = np.empty([n, 1, 400])
    for i in range(n):
        y = simulation_data(False,noise_level,False)
        clean[i, 0] = y
        noisy[i, 0] = y + g_noise(noise_level)
    return clean,noisy

