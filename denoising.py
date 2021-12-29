from model import GN
from utils import util
import os
import time
import torch
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from utils.Preprocess import calc_gradient_penalty
from utils.Data_generator import batch_simulation_data
from timeit import default_timer as timer
from sklearn.utils import shuffle
from utils.transformation import batch_transformation
from utils.transformation import transformation
from utils.Data_generator import test_simulation_data
from utils.plot import plot_one_signal
from utils.util import average_snr
import scipy.io as io
from glob import glob
# ---------------------Please Set the GPU id------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "*"

torch.set_default_tensor_type('torch.FloatTensor')
lrd = 0.0001  # set by yourself
batch_size = 128 # set by yourself
restore_mode = False # load trained model
start_epoch = 0
latest_generator_model = "*.pth"
latest_denoiser_model = "*.pth"
# we recommend to generate training set with *.npy
dataset_path_noisy = './*.npy'
dataset_path_clean = './*.npy'
# Note the name format for val dataset
dataset_val_clean = sorted(glob('./*/'+'clean*.npy'))
dataset_val_noisy = sorted(glob('./*/'+'noisy*.npy'))
# 0: training 1:test
mode = 0 #

test_path = '/*.npy'

train_epoch = 400
train_iters = 1000


name = "/*/TEM-NLnet"

if not os.path.exists(name):
    os.makedirs(name)
    print('-'*20,'create a new path','-'*20)

def train():
    # set the running device (gpu or cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    # accelerate using cuDNN
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True

    # ---------------------In denoising branch, denoiser needs to be initialized------------------------
    if restore_mode:
        denoiser = torch.load(latest_denoiser_model)
    else:
        denoiser = GN.TEMDnet()
    print('---------- Networks initialized -------------')
    util.print_network(denoiser)
    print('-----------------------------------------------')

    # optimizer
    optimizer_d = torch.optim.Adam(denoiser.parameters(), lr=lrd, betas=(0, 0.9))

    # transform to GPU

    denoiser.to(device)
    # set log
    writer = SummaryWriter(log_dir=name, flush_secs=60)
    # dataset split 8/1/1
    clean = np.load(dataset_path_clean)
    noisy = np.load(dataset_path_noisy)

    clean_val = []
    noisy_val = []
    for path in dataset_val_clean:
        clean_val.append(np.load(path))
    for path in dataset_val_noisy:
        noisy_val.append(np.load(path))

    clean, noisy = shuffle(clean, noisy)
    clean_train = clean[:int(clean.shape[0] * 0.8)]
    noisy_train = noisy[:int(clean.shape[0] * 0.8)]

    # activate denoiser
    for p in denoiser.parameters():  # reset requires_grad
        p.requires_grad_(True)

    for epoch in range(start_epoch, train_epoch):
        # print('Epoch: ' + str(epoch) + '/' + str(train_epoch))
        start_time = time.time()
        train_iters = clean_train.shape[0] // batch_size
        # shuffle dataset
        clean_train, noisy_train = shuffle(clean_train, noisy_train)
        print('---------- dataset shuffle -------------')
        for iter in range(train_iters):
            # print('Iter: ' + str(iter) + '/' + str(train_iters))
            start = timer()
            # grad set to zero
            optimizer_d.zero_grad()

            s = clean_train[(iter * batch_size):(iter * batch_size + batch_size)].reshape([batch_size, 1, 400])
            fake_data = noisy_train[(iter * batch_size):(iter * batch_size + batch_size)].reshape([batch_size, 1, 400])
            fake_data = batch_transformation(fake_data, size=20, batch_size=batch_size)
            fake_data = torch.tensor(fake_data, dtype=torch.float).to(device)
            fake_data.requires_grad_(True)

            # train
            denoising_result = denoiser(fake_data)

            # calculate loss
            real_data = s
            real_data = batch_transformation(real_data, size=20, batch_size=batch_size)
            real_data = torch.from_numpy(real_data).to(device)
            real_data = real_data.float()
            real_data.requires_grad_(True)
            loss_mse = nn.MSELoss(reduction='mean')
            loss = loss_mse(real_data, denoising_result)
            # bp
            loss.backward()
            # optimize
            optimizer_d.step()


            # ------------------VISUALIZATION----------
            if iter % 100 == 0:
                print('iter:{}/{},epoch:{}/{},loss:{}'.format(iter, train_iters, epoch, train_epoch, loss))
                writer.add_scalar('data/loss', loss, (iter + epoch * train_iters))
        if epoch % 5 == 0:

            # displace the result
            snr_before = []
            snr_denoising = []

            for j in range(2):
                temp_before = []
                temp_denoising = []
                for i in range(len(dataset_val_noisy)):

                    # start to test 400
                    test_sample_clean = clean_val[0]

                    fig = plt.subplot(111)
                    noisy_signal = np.reshape(test_sample_clean[0], [400, ])
                    t = np.arange(0, 400)
                    plt.plot(t, noisy_signal, label='before')
                    plt.legend(loc="upper right")
                    plt.show()

                    test_sample_noisy = noisy_val[i]

                    fake_data = batch_transformation(test_sample_noisy, size=20, batch_size=1000)
                    fake_data = torch.tensor(fake_data, dtype=torch.float).to(device)
                    fake_data.requires_grad_(False)

                    denoising_result = denoiser(fake_data).detach().cpu().numpy()
                    denoising_result = batch_transformation(denoising_result, 20, batch_size=1000)
                    denoising_result = np.reshape(denoising_result, [1000, 1, 400])

                    if i ==4:
                        fig = plt.subplot(211)
                        noisy_signal = np.reshape(test_sample_noisy[0],[400,])
                        t = np.arange(0, 400)
                        plt.plot(t, noisy_signal,label='before')
                        plt.legend(loc="upper right")
                        fig = plt.subplot(212)
                        out = np.reshape(denoising_result[0],[400,])
                        t = np.arange(0, 400)
                        plt.plot(t, out, label="denoising", markersize=20)
                        plt.legend(loc="upper right")
                        plt.show()


                    snr_before.append(average_snr(test_sample_clean, test_sample_noisy))
                    snr_denoising.append(average_snr(test_sample_clean, denoising_result))
            for  i in range(len(dataset_val_noisy)):
                snr_before[i] = (snr_before[i] + snr_before[i+len(dataset_val_noisy)])/2
                snr_before[i] = (snr_before[i] + snr_before[i + len(dataset_val_noisy)]) / 2


            torch.save(denoiser, os.path.join(name, 'Epoch' + str(epoch) + 'TEMDnet.pth'))


            for i in range(len(dataset_val_noisy)):
                print('SNR Denoising-{}:'.format((i+15)),'{:.2f}. '.format(snr_denoising[i]),end="")
            print('\n')
            for i in range(len(dataset_val_noisy)):
                print('SNR before-{}:'.format((i+15)),'{:.2f}. '.format(snr_before[i]),end="")
            # lrd *= 0.95
        # print('[%d/%d] - time: %.2f\n' % ((epoch + 1), train_epoch, time.time() - start_time,))
    # ----------------------Save model----------------------


def test():

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    denoiser = torch.load(latest_denoiser_model)
    denoiser.to(device)

    test_sample_noisy = np.load(test_path)
    D1 = test_sample_noisy.shape[0]

    test_sample_noisy = np.reshape(test_sample_noisy, [D1, 1, 400])
    fake_data = batch_transformation(test_sample_noisy, size=20, batch_size=D1)
    fake_data = torch.tensor(fake_data, dtype=torch.float).to(device)

    # test
    denoising_result = denoiser(fake_data).detach().cpu().numpy()
    denoising_result = np.reshape(denoising_result, [D1, 20, 20])
    denoising_result = np.reshape(batch_transformation(denoising_result, 20, D1), [D1, 400])
    # save
    io.savemat('./*/denoising_result.mat', {'denoising_result': denoising_result})
    print(denoising_result.shape)


if __name__ == '__main__':
    if mode == 0:
        train()
    else:
        test()

