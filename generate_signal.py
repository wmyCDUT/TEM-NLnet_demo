import torch
from model import GN
import numpy as np
from utils.Data_generator import batch_simulation_data
from utils.Data_generator import is_preprocess_batch
from matplotlib import pyplot as plt
from scipy import stats
import matplotlib.pylab as pylab
from tensorboardX import SummaryWriter
from utils.plot import plot_one_signal
params = {
        'axes.labelsize': '35',  # 轴上字
        'xtick.labelsize': '30',  # 轴图例
        'ytick.labelsize': '30',  # 轴图例
        'lines.linewidth': 4,  # 线宽
        'legend.fontsize': '35',  # 图例大小
        'figure.figsize': '40, 25'  # set figure size,长12，宽9
    }
pylab.rcParams.update(params)

### you should set the parameters of theoretical signals you want
k1_min=50000
k1_max=120000
k2_min=10
k2_max=40
b_min=1500
b_max=2000

#

latest_generator_model = "/your_path/*.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('-'*20,'load model','-'*20)
G = torch.load(latest_generator_model)
print('-'*20,'done!','-'*20)

G.to(device)

input_noise = np.random.rand(1, 1, 400)
clean_original = batch_simulation_data(1,label_n=False,noise_level = 0,is_preprocess=False)
# get condition vector, i.e. the first 50 smooth vector
condition = clean_original[:,:,0:50]
input_noise = np.concatenate((input_noise, condition), axis=2)
noise = torch.from_numpy(input_noise).to(device)
noise = noise.float()
noise = G(noise).detach()

noise = noise.cpu().numpy()
noise = np.array(noise).reshape([400,])
clean_original = np.reshape(clean_original,[400,])

real_data_path = '/*.npy'
Real_data = np.load(real_data_path)
#print(Real_data.shape)
# set by you
displace_number = 70
real_data = np.reshape(Real_data[displace_number,:],[1,1,400])
real_data_smooth_noisy = is_preprocess_batch(real_data, 1)

fig = plt.subplot(221)
plt.ylim([-2000,10000])

t = np.arange(0, 400)
real_noise = np.reshape(real_data_smooth_noisy,[400,])
plt.plot(t, real_noise,label="Actual noisy",markersize=20) #
plt.legend(loc="upper right")
plt.title('(a)', fontsize = 35, y=-0.18)
plt.ylabel('Amptitude(nT)')
plt.xlabel('t(ms)')
fig =plt.subplot(223)
plt.ylim([0,60000])
real_data = np.reshape(real_data,[400,])
t = np.arange(0, 400)

plt.plot(t, real_data,label="Actual Noisy TEM Signal",markersize=20) #
plt.legend(loc="upper right")
plt.title('(c)', fontsize = 35,y=-0.18)
plt.ylabel('Amptitude(nT)')
plt.xlabel('t(ms)')

fig = plt.subplot(222)
learn_noise = np.reshape(noise,[400,])
plt.ylim([-2000,10000])
t = np.arange(0, 400)

plt.plot(t, learn_noise,label="Learned Noise By Ours",markersize=20) #
plt.legend(loc="upper right")
plt.title('(b)', fontsize = 35,y=-0.18)
plt.ylabel('Amptitude(nT)')
plt.xlabel('t(ms)')
fig = plt.subplot(224)
plt.ylim([0,60000])
learn_noise = np.reshape(noise,[400,])
learn_TEM = clean_original + learn_noise
t = np.arange(0, 400)

plt.plot(t, learn_TEM,label="Learned Noisy TEM Signal By Ours",markersize=20) #
plt.legend(loc="upper right")
plt.title('(d)',fontsize = 35, y=-0.18)
plt.ylabel('Amptitude(nT)')
plt.xlabel('t(ms)')



plt.savefig('./*.pdf', dpi=250, bbox_inches='tight')
plt.show()





