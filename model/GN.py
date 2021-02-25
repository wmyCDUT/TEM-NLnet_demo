import torch.nn as nn
from model.basic import Conv_Block, Down_Conv, DSConv, Up_Conv, IRB, Dilation_conv_1,Dilation_conv_2,residual_block_v1,residual_block_v2


class Generator(nn.Module):
    # initializers
    def __init__(self):
        super(Generator, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(400, 600),
            nn.ReLU(True),
            nn.Linear(600, 600),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(600, 600),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(600, 600),
            nn.ReLU(True),
            nn.Linear(600, 400),
        )

    # forward method
    def forward(self, x):
        out = self.lin(x)
        return out


class Discriminator(nn.Module):
    # initializers
    def __init__(self, num_classes=1):
        super(Discriminator, self).__init__()
        self.classification = nn.Sequential(
            nn.Linear(400, 200),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(200, 100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100, 32),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(32, num_classes)
        )

    # forward method
    def forward(self, x):
        out = self.classification(x)
        return out
 
class TEMDnet(nn.Module):
    def __init__(self):
        super(TEMDnet,self).__init__()
        self.conv1 = Dilation_conv_1(1,32)
        self.conv2 = Dilation_conv_2(32,64)
        self.conv3 = residual_block_v1(64,128)
        self.conv4 = residual_block_v2(128,128)
        self.conv5 = residual_block_v2(128,128)
        self.conv6 = residual_block_v2(128,128)
        self.conv7 = residual_block_v2(128,128)
        self.conv8 = residual_block_v1(128,64)
        self.conv9 = Dilation_conv_2(64,32)
        self.conv10 = nn.Conv2d(32, 1, 3, padding=1, bias=False)   
    def forward(self,x):
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        return (x - out)

