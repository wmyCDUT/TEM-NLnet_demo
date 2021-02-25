from model import GN
from utils import util
import os
import time
import torch
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import *
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
from utils.util import average_snr
# ---------------------Please Set the GPU id------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

torch.set_default_tensor_type('torch.FloatTensor')
lrd = 0.001
batch_size = 128
restore_mode = False
start_epoch = 0
model_name = 'denoiser_real'
latest_generator_model = "/*.pth"
latest_denoiser_model = "/*.pth" 
dataset_path_noisy = '/*.npy'
dataset_path_clean = '/*.npy'
dataset_path_noisy_test_5 = '/*.npy'
dataset_path_clean_test_5 = '/*.npy'
train_epoch = 400
train_iters = 1000
label_k1 = 5 * 10000
label_k2 = 1
label_b = 1500
label_n = 200
gp_lambda = 10
name = "/*"


def train():
    # set the running device (gpu or cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    # accelerate using cuDNN 
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True

    # -------------------In denoising branch, trained Generator needs to be loaded for noise generation------------------------
    G = torch.load(latest_generator_model)
    print("finish load module")
    # ---------------------In denoising branch, denoiser needs to be initialized------------------------
    if restore_mode:
        denoiser = torch.load(latest_denoiser_model)
    else:
        denoiser = GN.TEMDnet()
    print('---------- Networks initialized -------------')
    util.print_network(denoiser)
    print('-----------------------------------------------')
    
    # optimizer
    #optimizer_d = torch.optim.Adam(denoiser.parameters(), lr=lrd, betas=(0, 0.9))
    optimizer_d = Adam(denoiser.parameters(),lr = lrd)
    scheduler = lr_scheduler.StepLR(optimizer_d,step_size=10,gamma = 0.95)
    # transform to GPU
    G.to(device)
    denoiser.to(device)
    # set log
    writer = SummaryWriter(log_dir=name, flush_secs=60)
    # dataset split 8/1/1
    clean = np.load(dataset_path_clean)
    noisy = np.load(dataset_path_noisy)
    clean_test_5 = np.load(dataset_path_clean_test_5)
    noisy_test_5 = np.load(dataset_path_noisy_test_5)
    clean,noisy = shuffle(clean,noisy)
    clean_train = clean[:int(clean.shape[0]*0.8)]
    noisy_train = noisy[:int(clean.shape[0]*0.8)]
    #print(clean_train.shape)
    # freeze G
    for p in G.parameters():  # reset requires_grad
                p.requires_grad_(False)
    # activate denoiser
    for p in denoiser.parameters():  # reset requires_grad
                p.requires_grad_(True)
               
    for epoch in range(start_epoch,train_epoch):
        # print('Epoch: ' + str(epoch) + '/' + str(train_epoch))
        start_time = time.time()
        train_iters = clean_train.shape[0]//batch_size
        # shuffle dataset
        clean_train,noisy_train = shuffle(clean_train,noisy_train)
        scheduler.step()
        print('---------- dataset shuffle -------------')
        for iter in range(train_iters):
            # print('Iter: ' + str(iter) + '/' + str(train_iters))
            start = timer()
            # grad set to zero
            optimizer_d.zero_grad()
            

            # gen fake data and load real data
            noise = np.random.rand(batch_size, 1, 400)
            noise = torch.from_numpy(noise).to(device)
            noise = noise.float()
            noise.requires_grad_(True)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
            noisev.requires_grad_(False)
            fake_noise = G(noisev).detach().cpu().numpy()
            s = clean_train[(iter*batch_size):(iter*batch_size+batch_size)].reshape([batch_size,1,400])
            fake_data = s + fake_noise
        
            fake_data = batch_transformation(fake_data,size=20, batch_size=batch_size)
            fake_data = torch.tensor(fake_data, dtype=torch.float).to(device)
            fake_data.requires_grad_(True)
            
            # train 
            denoising_result = denoiser(fake_data)
            
            # calculate loss
            real_data = s
            real_data = batch_transformation(real_data,size=20,batch_size=batch_size)
            real_data = torch.from_numpy(real_data).to(device)
            real_data = real_data.float()
            real_data.requires_grad_(True)
            loss_mse = nn.MSELoss(reduction='mean')
            loss = loss_mse(real_data,denoising_result)
            # bp
            loss.backward()
            # optimize
            optimizer_d.step()
            #if iter % 100 == 0:
            #    print(denoiser.state_dict()['conv10.weight'])
            end = timer()
            #print(f'---train D elapsed time: {end - start}')
            
            # ------------------VISUALIZATION----------
            if iter % 100 ==0:
                print('iter:{}/{},epoch:{}/{},loss:{},learning rate:{}'.format(iter,train_iters,epoch,train_epoch,loss,optimizer_d.state_dict()['param_groups'][0]['lr']))
                writer.add_scalar('data/loss', loss, (iter + epoch * train_iters))
        if epoch % 1 == 0:
            # start to test 400
            test_sample_clean = clean_test_5[:1000]
            test_sample_noisy = noisy_test_5[:1000]
            fake_data = batch_transformation(test_sample_noisy, size=20, batch_size=1000)
            fake_data = torch.tensor(fake_data, dtype=torch.float).to(device)
            fake_data.requires_grad_(False)
            
            denoising_result = denoiser(fake_data).detach().cpu().numpy()
            denoising_result = batch_transformation(denoising_result,20,batch_size=1000)
            denoising_result = np.reshape(denoising_result,[1000,1,400])
            
            snr_before = average_snr(test_sample_clean,test_sample_noisy)
            snr_denoising_5 = average_snr(test_sample_clean,denoising_result)
            
            
            
            torch.save(denoiser, os.path.join(name, 'Epoch' + str(epoch) + 'TEMDnet.pth'))
            print('snr denoising 5:{}'.format(snr_denoising_5))
            #lrd *= 0.95 
        # print('[%d/%d] - time: %.2f\n' % ((epoch + 1), train_epoch, time.time() - start_time,))
    # ----------------------Save model----------------------
    
    
    
def test():
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        denoiser = torch.load(latest_denoiser_model)
        denoiser.to(device)
        test_sample_clean, test_sample_noisy= test_simulation_data(1, noise_level=600)
        #np.save('test_clean2.npy',test_sample_clean)
        #np.save('test_noisy2.npy',test_sample_noisy)
        fake_data = batch_transformation(test_sample_noisy, size=20, batch_size=1)
        fake_data = torch.tensor(fake_data, dtype=torch.float).to(device)
        
        # test
        denoising_result = denoiser(fake_data).detach().cpu().numpy()
        denoising_result = np.reshape(denoising_result,[20,20])
        denoising_result = np.reshape(transformation(denoising_result,20),[1,400])
        np.save('denoising_result_test2.npy',denoising_result)
        print(denoising_result.shape)
        
if __name__ == '__main__':
    train()
    #test()

