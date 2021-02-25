import pandas as pd
import csv
import  numpy as np
import scipy.io as io
import os
from glob import  glob

SAVE_DIR = './toMat'
points = * # determined by yourself
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print('create SAVE DIR ')
files = glob('./*.dat')
i = 0
out = np.empty([3198,400])
row_num = 0
for file in files:
    # open file
    data = open(file, 'r')
    lines = data.readlines()
    len_ = len(lines)
    data = np.empty([len_,])
    i = 0
    for line in lines:
        data[i] = float(line.split('\t')[-1].strip('\n'))
        i += 1
    times = int(len_/40)
    for time in range(times):
        num = time
        for point in range(points):
            out[row_num, point] = data[num]
            num += times
            if point== (points-1):
                out[row_num,points:] = out[row_num,points-1]
        row_num+=1
# save
np.save('training_set.npy',out)
