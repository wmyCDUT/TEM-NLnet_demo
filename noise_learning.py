from model import GN
from utils import util
import os
import time
import torch
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorboardX import SummaryWriter

import torch.nn as nn
import torch.optim as optim
from torch.optim import *
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from utils.Preprocess import calc_gradient_penalty
from utils.Data_generator import batch_simulation_data
from utils.Data_generator import is_preprocess_batch
from utils.Data_generator import get_condition
from timeit import default_timer as timer
# ---------------------Please Set the GPU id------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "*"

torch.set_default_tensor_type('torch.FloatTensor')

# default setting
lrD = 0.0004
lrG = 0.0001
batch_size = 128
restore_mode = False
# reduce the effect of early-time signal
preprocess = True
start_epoch = 0
latest_generator_model = "/your_path/*.pth"
latest_discriminator_model = "/your_path/*.pth"
real_data_path = '/your_path/*.npy'
name = "noise_learning"
model_name = 'generate_noise'
if_plot = False

train_epoch = 201
train_iters = 1000

gen_iters = 1
critic_iters = 5

label_k1 = 5 * 10000
label_k2 = 1
label_b = 1500
label_n = 200
gp_lambda = 10

### you should set the parameters of theoretical signals you want
k1_min=50000
k1_max=120000
k2_min=10
k2_max=40
b_min=1500
b_max=2000


if not os.path.exists(name):
    os.makedirs(name)
    print('-'*20,'create a new path','-'*20)


def train():
    # gpu setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    # device = torch.device('cpu')
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True

    # network
    if restore_mode:
        G = torch.load(latest_generator_model)
        D = torch.load(latest_discriminator_model)
        print("finish load module")
    else:
        G = GN.Generator()
        D = GN.Discriminator()

        print("finish new module")
    
    print('---------- Networks initialized -------------')
    util.print_network(G)
    util.print_network(D)

    # load_real_data

    Real_data = np.load(real_data_path)
    D1 = Real_data.shape[0]
    print('-'*20,'load model done','-'*20)
    print('sample number:{}'.format(D1))

    
    print('-----------------------------------------------')
    optimizer_d = Adam(D.parameters(),lr = lrD)
    scheduler_d = lr_scheduler.StepLR(optimizer_d,step_size=10,gamma = 0.95)
    optimizer_g = Adam(G.parameters(),lr = lrG)
    scheduler_g = lr_scheduler.StepLR(optimizer_d,step_size=10,gamma = 0.95)


    G.to(device)
    D.to(device)
    writer = SummaryWriter(log_dir=name, flush_secs=60)

    for epoch in range(start_epoch, train_epoch):
        print('Epoch: ' + str(epoch) + '/' + str(train_epoch))
        start_time = time.time()
        Real_data = shuffle(Real_data)
        train_iters = D1//batch_size
        # D count_num for dataset
        count_num = 0
        for iter in range(train_iters):
            # ---------------------TRAIN D------------------------
            for p in D.parameters():  # reset requires_grad
                p.requires_grad_(True)  # they are set to False below in training G

            for i in range(critic_iters):
                #print("Critic iter: " + str(i))

                start = timer()
                optimizer_d.zero_grad()

                # gen fake data and load real data
                # get uniform noise for input
                noise = np.random.rand(batch_size, 1, 400)

                if count_num == train_iters:
                           count_num =0

                real_data_original = Real_data[(count_num * batch_size):(count_num * batch_size + batch_size)].reshape(
                    [batch_size, 1, 400])
                # get conditional vector, i.e. the first 50 smooth vectors.
                # *** conditional vector gives the extra information for noise generation, which can control the noise level.
                # *** if the signal amplitude is high, noise will be higher relatively
                condition = get_condition(real_data_original,batch_size)
                # concatenate input noise with conditional vector along with channel
                input_noise = np.concatenate((noise,condition),axis=2)
                input_noise = torch.from_numpy(input_noise).to(device)
                input_noise = input_noise.float()
                input_noise.requires_grad_(True)

                with torch.no_grad():
                    noisev = input_noise  # totally freeze G, training D
                noisev.requires_grad_(False)
                fake_data = G(noisev).detach()


                fake_data = fake_data.cpu().numpy()
                # here we wish the noise decouple with signal
                fake_data = torch.tensor(fake_data , dtype=torch.float).to(device)


                count_num +=1

                # reduce the effect of early-time signal
                real_data_smooth_noisy = is_preprocess_batch(real_data_original, batch_size)
                real_data_smooth = torch.from_numpy(real_data_smooth_noisy).to(device)
                real_data_smooth = real_data_smooth.float()
                real_data_smooth.requires_grad_(True)

                # train with real data
                disc_real = D(real_data_smooth)
                disc_real = disc_real.mean()

                # train with fake data
                disc_fake = D(fake_data)
                disc_fake = disc_fake.mean()
                # showMemoryUsage(0)
                # train with interpolates data
                gradient_penalty = calc_gradient_penalty(D, real_data_smooth, fake_data, batch_size, 400, device, gp_lambda)
                # showMemoryUsage(0)

                # final disc cost
                disc_cost = disc_fake - disc_real + gradient_penalty
                disc_cost.backward()
                #print(disc_cost.grad)
                w_dist = disc_real - disc_fake
                optimizer_d.step()
                end = timer()
                # print(f'---train D elapsed time: {end - start}')

                # ------------------VISUALIZATION----------
                if iter == 999 and i == (critic_iters-1):
                    writer.add_scalar('data/disc_cost', disc_cost, (iter + epoch * train_iters))
                    writer.add_scalar('data/gradient_pen', gradient_penalty, (iter + epoch * train_iters))
                    writer.add_scalar('data/w_cost', w_dist, (iter + epoch * train_iters))

            start = timer()
            # ---------------------TRAIN G------------------------
            for p in D.parameters():
                p.requires_grad_(False)  # freeze D

            gen_cost = None
            for i in range(gen_iters):
                #print("Generator iters: " + str(i))
                optimizer_g.zero_grad()

                noise = np.random.rand(batch_size, 1, 400)

                clean_original = batch_simulation_data(batch_size, label_n=False,noise_level=0,is_preprocess=False,k1_min=50000,k1_max=120000,k2_min=10,k2_max=40,b_min=1500,b_max=2000)
                # get condition vector, i.e. the first 50 smooth vector
                condition = clean_original[:,:,0:50]
                input_noise = np.concatenate((noise, condition), axis=2)
                input_noise = torch.from_numpy(input_noise).to(device)
                input_noise = input_noise.float()
                input_noise.requires_grad_(True)

                fake_data_g = G(input_noise)



                gen_cost = D(fake_data_g)
                gen_cost = gen_cost.mean()
                gen_cost = -gen_cost
                gen_cost.backward()
            optimizer_g.step()
            end = timer()
            if if_plot:
                if epoch % 5 == 0 and iter == 999:
                    #print('-----------------------display---------------------')
                    #

                    fig = plt.subplot(611)
                    out_noise = np.reshape(noise[0], [400, ])
                    t = np.arange(0, 400)
                    plt.title("input noise")
                    # plt.xlabel("t axis caption")
                    # plt.ylabel("y axis caption")
                    plt.plot(t, out_noise)

                    fig = plt.subplot(612)
                    out = np.reshape(fake_data_g[0].cpu().detach().numpy(), [400, ])
                    t = np.arange(0, 400)
                    #plt.title("learned noisy")
                    #plt.xlabel("t axis caption")
                    #plt.ylabel("y axis caption")
                    plt.plot(t, out,label="learned noisy",markersize=20)
                    plt.legend(loc="upper right")
                    fig = plt.subplot(613)
                    clean = np.reshape(clean_original[0], [400, ])
                    plus_noise = clean
                    #plt.title()
                    # plt.xlabel("t axis caption")
                    # plt.ylabel("y axis caption")
                    plt.plot(t, plus_noise,label="simulated clean TEM signal",markersize=20)
                    plt.legend(loc="upper right")  # set legend location
                    #plt.show()
                    fig = plt.subplot(614)
                    clean = np.reshape(clean_original[0],[400,])
                    plus_noise = clean+out
                    #plt.title("learned TEM signal (figure1 + figure2)")
                    #plt.xlabel("t axis caption")
                    #plt.ylabel("y axis caption")
                    plt.plot(t, plus_noise,label='learned TEM signal (figure2 + figure3)',markersize=20)
                    plt.legend(loc="upper right")  # set legend location

                    fig = plt.subplot(615)
                    actual = np.reshape(Real_data[4],[400,])
                    #plt.title()
                    #plt.xlabel("t axis caption")
                    #plt.ylabel("y axis caption")
                    plt.plot(t, actual,label="actual noisy TEM signal",markersize=20)
                    plt.legend(loc="upper right")  # set legend location

                    fig = plt.subplot(616)
                    actual = np.reshape(real_data_smooth_noisy[0], [400, ])
                    # plt.title()
                    # plt.xlabel("t axis caption")
                    # plt.ylabel("y axis caption")
                    plt.plot(t, actual, label="actual noisy ", markersize=20)
                    plt.legend(loc="upper right")  # set legend location
                    plt.show()

                    #print('-----------------------display---------------------')
                # print(f'---train G elapsed time: {end - start}')

            # ---------------VISUALIZATION---------------------
            writer.add_scalar('data/gen_cost', gen_cost, (iter + epoch * train_iters))
            #print('gen cost:{}'.format(gen_cost))
            if iter == 999 :
                 print('iter:{}/{},epoch:{}/{},gen_cost:{}'.format(iter,train_iters,epoch,train_epoch,gen_cost))    
                  
        if epoch % 10 == 0 and epoch != 0:
            torch.save(G, os.path.join(name, 'Epoch' + str(epoch) + 'generator_param'+model_name+'_real.pth'))
            torch.save(D, os.path.join(name, 'Epoch' + str(epoch) + 'discriminator_param'+model_name+'_real.pth'))

        scheduler_d.step()
        scheduler_g.step()
        #print('[%d/%d] - time: %.2f\n' % ((epoch + 1), train_epoch, time.time() - start_time,))
    # ----------------------Save model----------------------
    torch.save(G, os.path.join(name, 'generator_param2.pth'))
    torch.save(D, os.path.join(name, 'discriminator_param2.pth'))



if __name__ == '__main__':
    train()


