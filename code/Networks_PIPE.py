import numpy as np
import torch.nn as nn
from torch.nn.functional import interpolate
import torch
import torch.optim as optim
import copy
import math
from BatchMaker import *
smaller_cube = False
crop = 8
EPS = 10e-5
modes = ['bilinear', 'trilinear']
from torch.distributed.pipeline.sync import Pipe

def return_D_nets(ngpu, n_dims, device, lr, beta1, anisotropic, D_images, scale_f, rotation, rotation_bool):
    """
    :return: Returns the Batch Makers, Discriminators, and Optimizers for the Discriminators. If the material is isotropic, there is 1 Discriminator, 
    just repeated 3 times.If the material is anisotropic, there are 3 different Discriminators.
    """
    D_nets = []
    D_optimisers = []
    D_BMs = []
    if anisotropic:
        for i in np.arange(n_dims):
            BM_D = BatchMaker(device, path=D_images[i], sf=scale_f, dims=n_dims, stack=True, low_res=False, rot_and_mir=rotation_bool[i])
            nc_d = len(BM_D.phases)
            # Create the Discriminator
            netD = Discriminator3d(nc_d).to(device)
            # Handle multi-gpu if desired
            if (device.type == 'cuda') and (ngpu > 1):
                netD = nn.DataParallel(netD, list(range(ngpu)))
            optimiserD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
            # append the BMs, nets and optimisers:
            D_BMs.append(BM_D)
            D_nets.append(netD)
            D_optimisers.append(optimiserD)

    else:  # material is isotropic
        # Create the batch maker
        BM_D = BatchMaker(device, path=D_images[0], sf=scale_f, dims=n_dims, stack=True, low_res=False, rot_and_mir=rotation)
        # Create the Discriminator
        nc_d = len(BM_D.phases)
        netD = Discriminator3d(nc_d).to(device)
        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netD = nn.DataParallel(netD, list(range(ngpu)))
        optimiserD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        D_BMs = [BM_D]*n_dims  # same batch maker, different pointers
        D_nets = [netD]*n_dims  # same network, different pointers
        D_optimisers = [optimiserD]*n_dims  # same optimiser, different pointers
    return D_BMs, D_nets, D_optimisers

# Generator Code
class Generator3D(nn.Module):
    def __init__(self, nc_g, nc_d, scale_factor):
        super(Generator3D, self).__init__()
        ### Layers below will be distributed into GPU 1 if DPP###
        self.conv_1 = nn.Conv3d(nc_g, 512, 3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm3d(512)
        self.ReLU_1 = nn.ReLU()
        self.ResBlock_1 = ResidualBlock(nn.Conv3d(512, 512, 3, stride=1, padding=1), 
                                        nn.BatchNorm3d(512), 
                                        nn.ReLU(), 
                                        nn.Conv3d(512, 512, 3, stride=1, padding=1),
                                        nn.BatchNorm3d(512))
        self.Upsample_1 = nn.Upsample(scale_factor=2, mode=modes[1])
        self.conv_2 = nn.Conv3d(512, 256, 3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm3d(256)
        self.ReLU_2 = nn.ReLU()
        ### Layers below will be distributed into GPU 2 if DPP###
        self.trans_1 = nn.ConvTranspose3d(256, 128, 4, 2, 1)
        self.bn_trans_1 = nn.BatchNorm3d(128)
        self.ReLU_3 = nn.ReLU()
        self.conv_3 = nn.Conv3d(128, 64, 3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm3d(64)
        self.ReLU_4 = nn.ReLU()
        ### Layers below will be distributed into GPU 3 if DPP###
        self.Upsample_2 = nn.Upsample(scale_factor=2, mode=modes[1])
        self.conv_4 = nn.Conv3d(64, 32, 3, stride=1, padding=1) 
        self.bn_4 = nn.BatchNorm3d(32)
        self.ReLU_5 = nn.ReLU()
        self.conv_end = nn.Conv3d(32, nc_d, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.ReLU_1(x)
        x = self.ResBlock_1(x)
        x = self.Upsample_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.ReLU_2(x)
        x = self.trans_1(x)
        x = self.bn_trans_1(x)
        x = self.ReLU_3(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.ReLU_4(x)
        x = self.Upsample_2(x)
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.ReLU_5(x)
        x = self.conv_end(x)
        x = nn.Softmax(dim=1)(x)
        return x

class ResidualBlock(nn.Module):
        def __init__(self, *layers):
            super(ResidualBlock, self).__init__()
            self.layers = nn.Sequential(*layers)

        def forward(self, x):
            residual = x
            out = self.layers(x)
            out += residual  # Add the residual connection
            return nn.ReLU()(out)
            
# Distribute the 3D Generator model into 3 GPUs with the Pipleline Parallel
def G_PP(device, nc_g, nc_d, scale_factor):
    
    netG = Generator3D(nc_g, nc_d, scale_factor).to(device)

    netG_stage1 = nn.Sequential(
        netG.conv_1, 
        netG.bn_1, 
        nn.ReLU(),
        ResidualBlock(nn.Conv3d(512, 512, 3, stride=1, padding=1), nn.BatchNorm3d(512), nn.ReLU(), nn.Conv3d(512, 512, 3, stride=1,
                                                                                                             padding=1),nn.BatchNorm3d(512)),
        nn.Upsample(scale_factor=2, mode=modes[1]),
        netG.conv_2,
        netG.bn_2,
        nn.ReLU(),
        )
    
    netG_stage2 = nn.Sequential(
        netG.trans_1,
        netG.bn_trans_1,
        nn.ReLU(),
        netG.conv_3,
        netG.bn_3,
        nn.ReLU(),
        )
    
    netG_stage3 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode=modes[1]),
        netG.conv_4,
        netG.bn_4,
        nn.ReLU(),
        netG.conv_end,
        nn.Softmax(dim=1),
        )
    
    device1 = torch.device("cuda:0")
    device2 = torch.device("cuda:1")
    device3 = torch.device("cuda:2")
    
    netG_stage1.to(device1)
    netG_stage2.to(device2)
    netG_stage3.to(device3)
    
    model_G = nn.Sequential(netG_stage1, netG_stage2, netG_stage3)
    
    model_G = Pipe(model_G, chunks=2)
    return model_G

# Discriminator code
class Discriminator3d(nn.Module):
    def __init__(self, nc_d):
        super(Discriminator3d, self).__init__()
        self.conv2 = nn.Conv2d(nc_d, 16, 4, 2, 1)
        self.conv3 = nn.Conv2d(16, 32, 4, 2, 1)
        self.conv4 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv5 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv6 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv7 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv8 = nn.Conv2d(512, 1, 3, 2, 0)

    def forward(self, x):
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        x = nn.ReLU()(self.conv5(x))
        x = nn.ReLU()(self.conv6(x))
        x = nn.ReLU()(self.conv7(x))
        return self.conv8(x)