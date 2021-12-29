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

def get_condition(input,n):
          out = np.empty([n, 1, 50])
          for i in range(n):
                 y = smooth(input[i,0,:],WINDOW)[0:50]
                 out[i, 0] = y
          return out

def simulation_data(sigma,noise_level,preprocess,k1_min=50000,k1_max=120000,k2_min=10,k2_max=40,b_min=1500,b_max=2000):
    t = np.linspace(0, 4, 400)
    k1 = np.random.randint(k1_min, k1_max)

    k2 = np.random.randint(k2_min,k2_max)

    b = np.random.randint(b_min, b_max)
    y = k1 * np.exp(-k2 * t)  + b
    if sigma is True:
        noise = g_noise(noise_level)
        out = y + noise
        if preprocess:
                 out = is_preprocess(out)
    else:
        out = y
    return out

def batch_simulation_data(n, label_n,noise_level,is_preprocess,k1_min=50000,k1_max=120000,k2_min=10,k2_max=40,b_min=1500,b_max=2000):
    out = np.empty([n, 1, 400])
    for i in range(n):
        y = simulation_data(label_n,noise_level,is_preprocess,k1_min=k1_min,k1_max=k1_max,k2_min=k2_min,k2_max=k2_max,b_min=b_min,b_max=b_max)
        out[i, 0] = y
    return out

def batch_simulation_single(n, label_n,noise_level,is_preprocess):
    out = np.empty([n, 1, 400])
    for i in range(n):
        t = np.linspace(0, 4, 400)
        k1 = 70000
        # k1 = np.random.randint(400, 500)
        k2 = 20
        # b = np.random.randint(1450, 1550)
        b = 1650
        y = k1 * np.exp(-k2 * t) + b
        out[i, 0] = y
    # out = np.maximum(out, 0.01)
    # out = np.log(out)
    return out

def test_simulation_data(n, noise_level):
    clean = np.empty([n, 1, 400])
    noisy = np.empty([n, 1, 400])
    for i in range(n):
        y = simulation_data(False,noise_level,False)
        clean[i, 0] = yc
        noisy[i, 0] = y + g_noise(noise_level)
    return clean,noisy



