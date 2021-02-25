from model import GN
from utils import util
import os
import time
import torch
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
from timeit import default_timer as timer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.set_default_tensor_type('torch.FloatTensor')
model_name = 'generate_noise_proprecess'
lrD = 0.0001
lrG = 0.0001
batch_size = 128
restore_mode = False
preprocess = True
start_epoch = 0
latest_generator_model = "/*.pth"
latest_discriminator_model = "/*.pth"
real_data_path = '/*.npy'


train_epoch = 40
train_iters = 1000
gen_iters = 1
critic_iters = 5
label_k1 = 5 * 10000
label_k2 = 1
label_b = 1500
label_n = 200
gp_lambda = 10
name = "/*"


def train():
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
    print('sample number:{}'.format(D1))

    
    print('-----------------------------------------------')
    optimizer_d = Adam(D.parameters(),lr = lrD)
    scheduler_d = lr_scheduler.StepLR(optimizer_d,step_size=10,gamma = 0.95)
    optimizer_g = Adam(G.parameters(),lr = lrG)
    scheduler_g = lr_scheduler.StepLR(optimizer_d,step_size=10,gamma = 0.95)
    
    #optimizer_g = torch.optim.Adam(G.parameters(), lr=lrd, betas=(0, 0.9))
    #optimizer_d = torch.optim.Adam(D.parameters(), lr=lrD, betas=(0, 0.9))

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
            # print('Iter: ' + str(iter) + '/' + str(train_iters))

            # ---------------------TRAIN D------------------------
            for p in D.parameters():  # reset requires_grad
                p.requires_grad_(True)  # they are set to False below in training G

            for i in range(critic_iters):
                #print("Critic iter: " + str(i))

                start = timer()
                optimizer_d.zero_grad()

                # gen fake data and load real data
                noise = np.random.rand(batch_size, 1, 400)
                noise = torch.from_numpy(noise).to(device)
                noise = noise.float()
                noise.requires_grad_(True)

                with torch.no_grad():
                    noisev = noise  # totally freeze G, training D
                noisev.requires_grad_(False)
                fake_data = G(noisev).detach()
                if preprocess:
                       fake_data = fake_data.cpu().numpy()
                s = batch_simulation_data(batch_size, label_n=False,noise_level=0,is_preprocess=False)
                if preprocess:
                      fake_data = is_preprocess_batch(fake_data + s,batch_size)
                      fake_data = torch.tensor(fake_data , dtype=torch.float).to(device)
                else:
                      fake_data = fake_data + torch.tensor(s, dtype=torch.float).to(device)
                
                if count_num == train_iters:
                           count_num =0
                real_data = Real_data[(count_num*batch_size):(count_num*batch_size+batch_size)].reshape([batch_size,1,400])
                count_num +=1
                if preprocess:
                       real_data= is_preprocess_batch(real_data,batch_size)
                real_data = torch.from_numpy(real_data).to(device)
                real_data = real_data.float()
                real_data.requires_grad_(True)

                # train with real data
                disc_real = D(real_data)
                disc_real = disc_real.mean()

                # train with fake data
                disc_fake = D(fake_data)
                disc_fake = disc_fake.mean()
                # showMemoryUsage(0)
                # train with interpolates data
                gradient_penalty = calc_gradient_penalty(D, real_data, fake_data, batch_size, 400, device, gp_lambda)
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
                if i == critic_iters - 1:
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
                noise = torch.from_numpy(noise).to(device)
                noise = noise.float()
                noise.requires_grad_(True)

                fake_data = G(noise)
                if iter % 200 == 0:
                    print('-----------------------noise---------------------')
                    print( fake_data[0])
                    print('-----------------------noise---------------------')
                if not preprocess:
                       s = batch_simulation_data(batch_size, label_n=False,noise_level=0,is_preprocess=False)
                       fake_data = fake_data + torch.tensor(s, dtype=torch.float).to(device)

                gen_cost = D(fake_data)
                gen_cost = gen_cost.mean()
                gen_cost = -gen_cost
                gen_cost.backward()
            optimizer_g.step()
            end = timer()
            # print(f'---train G elapsed time: {end - start}')

            # ---------------VISUALIZATION---------------------
            writer.add_scalar('data/gen_cost', gen_cost, (iter + epoch * train_iters))
            #print('gen cost:{}'.format(gen_cost))
            if iter % 20 == 0 :
                 print('iter:{}/{},epoch:{}/{},gen_cost:{}'.format(iter,train_iters,epoch,train_epoch,gen_cost))    
                  
        if epoch % 1 == 0:
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


