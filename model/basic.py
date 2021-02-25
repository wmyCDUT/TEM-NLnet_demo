import torch
import utils
import torch.nn as nn
import torch.nn.functional as F


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(Conv_Block, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.Conv(x)



class DSConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(DSConv, self).__init__()
        self.Conv1 = torch.nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channel, bias=True)
        # self.IN = nn.BatchNorm2d(in_channel)
        self.LRelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.Conv2 = Conv_Block(in_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.Conv1(x)
        # out = self.IN(out)
        out = self.LRelu(out)
        out = self.Conv2(out)
        return out
        
       
class Dilation_conv_1(nn.Module):
        def __init__(self,in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
            super(Dilation_conv_1,self).__init__()
            self.Conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
            self.Relu = nn.ReLU(True)
        def forward(self,x):
            out = self.Conv1(x)
            out = self.Relu(out)
            return out
       
class Dilation_conv_2(nn.Module):
        def __init__(self,in_channel, out_channel, kernel_size=3, stride=1, padding=2, dilation=2, bias=False):
            super(Dilation_conv_2,self).__init__()
            self.Conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
            self.Relu = nn.ReLU(True)
        def forward(self,x):
            out = self.Conv1(x)
            out = self.Relu(out)
            return out       
def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)
    
def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 1, stride=stride, padding=0, bias=False)    

class residual_block_v1(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block_v1,self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2
        
        self.pre = conv1x1(in_channel, out_channel)
        self.bn_pre = nn.BatchNorm2d(out_channel)
        
        self.conv1 = conv1x1(in_channel, in_channel)
        self.bn1 = nn.BatchNorm2d(in_channel)
        
        self.conv2 = conv3x3(in_channel, in_channel)
        self.bn2 = nn.BatchNorm2d(in_channel)
        
        self.conv3 = conv1x1(in_channel, out_channel)
        self.bn3 = nn.BatchNorm2d(out_channel)
        
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
    def forward(self, x):
        
        residual = self.pre(x)
        residual = self.bn_pre(residual)
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        out = self.conv3(out)
        out = self.bn3(out)

        return F.relu(residual+out, True)


class residual_block_v2(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block_v2,self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2
        
        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)

        return F.relu(x+out, True)
        


class IRB(nn.Module):
    def __init__(self):
        super(IRB, self).__init__()
        self.Conv1 = Conv_Block(64, 128, 1, 1, 0)
        self.Conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, groups=128, bias=True)
        # self.BN1 = nn.BatchNorm2d(128)
        self.LRelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.Conv3 = nn.Conv2d(128, 64, 1, 1)
        # self.BN2 = nn.BatchNorm2d(64)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Conv2(out)
        # out = self.BN1(out)
        out = self.LRelu(out)
        out = self.Conv3(out)
        # out = self.BN2(out)
        return out + x


class Down_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down_Conv, self).__init__()
        self.Conv1 = DSConv(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
        self.Conv2 = DSConv(in_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = self.Conv1(x)
        out2 = F.interpolate(x, scale_factor=0.5)
        out2 = self.Conv2(out2)
        return out1 + out2


class Up_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up_Conv, self).__init__()
        self.Conv1 = DSConv(in_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.Conv1(out)
        return out

