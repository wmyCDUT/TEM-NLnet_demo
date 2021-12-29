import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def snr_cal(clean,noisy):
      residual = np.mean(np.square(clean)) / np.mean(np.square(clean - noisy))
      snr = 10*np.log10(residual)
      return snr,residual
def average_snr(clean,noisy,index=0):
       sum_ = 0
       n = clean.shape[0]
       for i in range(n):
              snr,_ = snr_cal(np.array(clean[i]),np.array(noisy[i]))
              sum_ += snr
       if index == 4:
           print(sum_)
           print(clean[5][0][1:5])
           print(noisy[5][0][1:5])
           a,b = snr_cal(clean[5], noisy[5])
           print(b)

           a, b = snr_cal(clean[5], noisy[5])
           print(b)


       return (sum_/n)

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


