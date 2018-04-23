import os, time, sys
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import find_patches


class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self, in_c=2,d=32):
        super(Generator, self).__init__()
        conv1=nn.Conv2d(in_channels=in_c,out_channels=d,kernel_size=5,dilation=1,padding=0,stride=1)
        bn1=nn.BatchNorm2d(d)
        conv2=nn.Conv2d(in_channels=d,out_channels=d*2,kernel_size=3,dilation=1,padding=0,stride=2)
        bn2=nn.BatchNorm2d(d*2)
        conv3=nn.Conv2d(in_channels=d*2,out_channels=d*2,kernel_size=3,dilation=2,padding=0,stride=1)
        bn3=nn.BatchNorm2d(d*2)
        conv4=nn.Conv2d(in_channels=d*2,out_channels=d*4,kernel_size=3,dilation=3,padding=0,stride=1)
        bn4=nn.BatchNorm2d(d*4)
        conv5=nn.Conv2d(in_channels=d*4,out_channels=d*4,kernel_size=3,dilation=2,padding=0,stride=2)
        bn5=nn.BatchNorm2d(d*4)
        deconv1=nn.ConvTranspose2d(in_channels=d*4,out_channels=d*4,kernel_size=3,dilation=4,padding=0,stride=2)
        bn6=nn.BatchNorm2d(d*4)
        deconv2=nn.ConvTranspose2d(in_channels=d*4,out_channels=d*4,kernel_size=3,dilation=2,padding=0,stride=2)
        bn7=nn.BatchNorm2d(d*4)
        deconv3=nn.ConvTranspose2d(in_channels=d*4,out_channels=d*2,kernel_size=3,dilation=3,padding=0,stride=1)
        bn8=nn.BatchNorm2d(d*2)
        deconv4=nn.ConvTranspose2d(in_channels=d*2,out_channels=d,kernel_size=3,dilation=2,padding=0,stride=1)
        bn9=nn.BatchNorm2d(d)
        deconv5=nn.ConvTranspose2d(in_channels=d,out_channels=1,kernel_size=3,dilation=2,padding=0,stride=1)
        pad = nn.ReflectionPad2d((1,0,1,0))     
        leaky_relu=nn.LeakyReLU(2e-2)
        self.encoder=nn.Sequential(conv1,bn1,leaky_relu,conv2,bn2,leaky_relu,conv3,bn3,leaky_relu,conv4,bn4,leaky_relu,conv5,bn5)
        self.decoder=nn.Sequential(deconv1,bn6,leaky_relu,deconv2,bn7,leaky_relu,deconv3,bn8,leaky_relu,deconv4,bn9,leaky_relu,deconv5,pad)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,inp):
        inp=self.encoder(inp)
        op=self.decoder(inp)
        return op.clamp(0,1)


class GlobalDiscriminator(nn.Module):
    def __init__(self, in_c=1,d=32):
        super(GlobalDiscriminator, self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_c,out_channels=d,kernel_size=5,dilation=1,padding=0,stride=2)
        self.conv2=nn.Conv2d(in_channels=d,out_channels=d*2,kernel_size=3,dilation=1,padding=0,stride=2)
        self.conv3=nn.Conv2d(in_channels=d*2,out_channels=d*2,kernel_size=3,dilation=1,padding=0,stride=2)
        self.conv4=nn.Conv2d(in_channels=d*2,out_channels=d,kernel_size=3,dilation=1,padding=0,stride=2)
        self.leaky_relu=nn.LeakyReLU(2e-2)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,inp):
        x=self.leaky_relu(self.conv1(inp))
        x=self.leaky_relu(self.conv2(x))
        x=self.leaky_relu(self.conv3(x))
        x=self.leaky_relu(self.conv4(x))
        return x.view(x.size(0),-1)

class LocalDiscriminator(nn.Module):
    def __init__(self, in_c=1,d=32):
        super(LocalDiscriminator, self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_c,out_channels=d,kernel_size=5,dilation=1,padding=0,stride=2)
        self.conv2=nn.Conv2d(in_channels=d,out_channels=d*2,kernel_size=3,dilation=1,padding=0,stride=2)
        self.conv3=nn.Conv2d(in_channels=d*2,out_channels=d,kernel_size=3,dilation=1,padding=0,stride=2)
        self.leaky_relu=nn.LeakyReLU(2e-2)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,inp):
        x=self.leaky_relu(self.conv1(inp))
        x=self.leaky_relu(self.conv2(x))
        x=self.leaky_relu(self.conv3(x))
        return x.view(x.size()[0],-1)



class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self,in_c):
        super(Discriminator, self).__init__()
        self.globalDiscriminator = GlobalDiscriminator(in_c)
        self.localDiscriminator= LocalDiscriminator(in_c)
        self.fc1=nn.Linear(256,64)
        self.fc2=nn.Linear(64,1)
        if in_c>1:
            self.IS_LABEL_ADDED=True
        else:
            self.IS_LABEL_ADDED=False


    def forward(self,image,mask):
        img_patch,mask_patch=find_patches(image,mask,IS_LABEL_ADDED=self.IS_LABEL_ADDED)
        # print (img_patch.size())
        global_d_op=self.globalDiscriminator(image)
        local_d_op=self.localDiscriminator(img_patch)
        x=torch.cat((global_d_op,local_d_op),dim=1)
        x=self.fc2(self.fc1(x))
        return F.sigmoid(x)#x.clamp(-1,1)

        

class Classifier(nn.Module):
    # initializers
    def __init__(self, d=16):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 3, 2, 1, 0)
        self.fc=nn.Linear(27,10)
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5(x))
        # print(x.size())
        x = x.view(x.size(0),-1)
        # print(x.size())
        return F.softmax(self.fc(x))


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
