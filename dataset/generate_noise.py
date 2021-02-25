import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
def g_noise(sigma):
    p = np.random.randn(1, 400) * sigma
    print(p)
    return p
data = np.load('../*.npy') # load theoretical TEM signal
D1,D2 = data.shape
# add
for i in range(D1):
    noise = g_noise(sigma = None) # level of noise is determined by yourself
    data[i] = data[i] + noise
    # plot test
    if i % 1000 == 0:
        plt.figure()
        x = np.arange(0,400)
        plt.plot(x,np.reshape(data[i],[400,]))
        plt.show()
# save 
io.savemat('../noisy_TEM.mat',{'noisy_TEM':data})
np.save('../noisy_TEM.npy',data)
