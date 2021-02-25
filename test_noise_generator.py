import torch
from model import GN
import numpy as np
from utils.Data_generator import batch_simulation_data
from matplotlib import pyplot as plt
#from scipy import stats
from tensorboardX import SummaryWriter


def decode(x):
    x = x.reshape(20, 20)
    for k in range(1, 20, 2):
        x[k, :] = x[k, ::-1]
    x = x.reshape(400)
    return x



latest_generator_model = "/*.pth"
latest_discriminator_model = "/*.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# G = GN.Generator()
G = torch.load(latest_generator_model)
# D = GN.Discriminator()
D = torch.load(latest_discriminator_model)

G.to(device)
D.to(device)


noise = np.random.rand(1, 1, 400)
noise = torch.from_numpy(noise).to(device)
noise = noise.float()
noise = G(noise).detach()
print(noise)
s = batch_simulation_data(1, label_n=False,noise_level = 0,is_preprocess=False)
# noise = noise + torch.tensor(s, dtype=torch.float).to(device)
noise = noise.cpu().numpy()
noise = np.array(noise)
np.save('*.npy', noise)





