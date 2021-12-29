
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci
TEST = False
NUM = 1000# 128000

import os

### you should set the parameters of theoretical signals you want
k1_min=50000
k1_max=120000
k2_min=10
k2_max=40
b_min=1500
b_max=2000

def g_noise(sigma=100):
    #sigma = np.random.randint(4)#sigma = 3
    #sigma = SIGMA
    p = np.random.randn(1, 400) * sigma
    return p


def simulate_dataset(sigma=100):
    t = np.linspace(0, 4, 400)
    k1 = np.random.randint(k1_min, k1_max)
    ### it is a example when k2_min=10, k2_max=40
    k2 = [ ]
    k2_temp1 = np.random.randint(10, 25)
    k2 += [k2_temp1]
    k2_temp2 = np.random.randint(30, 40)
    k2 += [k2_temp2]
    num = np.random.randint(0,2)
    b = np.random.randint(b_min, b_max)
    #b = np.random.randint(1450, 1550)
    y = k1 * np.exp(-k2[num] * t) + b
    if sigma:
        noise = g_noise(sigma=sigma)
        out = y + noise
    else:
        out = y
    return y,out

def simulate_dataset_test(sigma):
    t = np.linspace(0, 4, 400)
    k1 = np.random.randint(k1_min, k2_max)
    # this range is not included in training stage, which can explore the generaliztion
    k2 = np.random.randint(25, 30)
 
    b = np.random.randint(b_min, b_max)
    y = k1 * np.exp(-k2 * t) + b
    if sigma:
        noise = g_noise(sigma=sigma)
        out = y + noise
    else:
        out = y
    return y,out

sigma_list = list(np.linspace(100,900,9))

for sigma in sigma_list:
    print('-'*20,'sigma:'+str(sigma),'-'*20)
    dataset_noisy = np.empty([NUM ,1,400])  #128000
    dataset_clean = np.empty([NUM ,1,400])

    for i in range(NUM):
        print(i)
        sigma_index = np.random.randint(0, 9)
        #
        if TEST:
              clean,noisy = simulate_dataset_test(sigma=sigma) #simulate_dataset(True)
        else:
              clean,noisy = simulate_dataset(sigma=sigma)
        dataset_clean[i, 0] = clean
        dataset_noisy[i, 0] = noisy
    np.save('../your_path/*.npy'.format(int(sigma)),dataset_noisy)
    np.save('../your_path/*.npy'.format(int(sigma)),dataset_clean)

    
    
    
    
    
    
