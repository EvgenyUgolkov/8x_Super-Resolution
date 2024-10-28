import BatchMaker
import LearnTools
import Networks_PIPE
import ImageTools
import argparse
import torch
import numpy as np
from tifffile import imsave, imread, imwrite
import torch.nn as nn
modes = ['bilinear', 'trilinear']
from torch.distributed.pipeline.sync import Pipe
import torch.distributed as dist
print('1')

# Initialize the distributed backend
dist.init_process_group(backend='nccl')
torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
print('2')

# Parsing arguments:
parser = argparse.ArgumentParser()

args = LearnTools.return_args(parser)
print('3')

progress_dir, wd, wg = args.directory, args.widthD, args.widthG
n_res_blocks, pix_distance = args.n_res_blocks, args.pixel_coefficient_distance
num_epochs, g_update, n_dims = args.num_epochs, args.g_update, args.n_dims
squash, down_sample = args.squash_phases, args.down_sample
D_dimensions_to_check, scale_f = args.d_dimensions_to_check, args.scale_factor
size_to_evaluate, separator = args.volume_size_to_evaluate, args.separator
g_file_name, super_sample = args.g_image_path, args.super_sampling
phases_to_low, g_epoch_id = args.phases_low_res_idx, args.g_epoch_id
print('4')

progress_main_dir = 'progress/' + progress_dir
# progress_main_dir = 'progress'
path_to_g_weights = progress_main_dir + '/g_weights' + g_epoch_id + '.pth'
# path_to_g_weights = progress_main_dir + '/g_weights_large_slice.pth'
G_image_path = 'data/' + g_file_name
# G_image_path = 'data/new_vol_down_sample.tif'
print('5')

rand_id = str(np.random.randint(10000))
print('6')

file_name = 'generated_tif' + rand_id + '.tif'
crop_to_cube = False
input_with_noise = True
all_pore_input = False
print('7')

# crop the edges
crop = 1

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 3

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print('device is ' + str(device))

# the material indices to low-res:
to_low_idx = torch.LongTensor(phases_to_low).to(device)

# Number of channels in the training images. For color images this is 3
if input_with_noise:
    nc_g = 1 + to_low_idx.size()[0] + 1 # channel for pore plus number of
else:
    nc_g = 1 + to_low_idx.size()[0]

nc_d = 4  

netG = Networks_PIPE.generator(ngpu, wg, nc_g, nc_d, n_res_blocks, n_dims, scale_factor=scale_f).to(device)

class ResidualBlock(nn.Module):
    def __init__(self, *layers):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.layers(x)
        out += residual  # Add the residual connection
        return nn.ReLU()(out)

netG_stage1 = nn.Sequential(
    netG.conv_1, 
    netG.bn_1, 
    nn.ReLU(),
    ResidualBlock(nn.Conv3d(512, 512, 3, stride=1, padding=1), nn.BatchNorm3d(512), nn.ReLU(), nn.Conv3d(512, 512, 3, stride=1, padding=1),   nn.BatchNorm3d(512)),
    
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

G_net = Pipe(model_G, chunks=2)

# G_net.load_state_dict(torch.load(path_to_g_weights, map_location=torch.device(device)))
G_net.load_state_dict(torch.load(path_to_g_weights))

# If the whole network is saved:
# G_net = torch.load(path_to_g_weights)

G_net.eval()

with torch.no_grad():  # save the images
    # 1. Start a new run
    # wandb.init(project='SuperRes', name='making large volume',
    #            entity='tldr-group')

#    step_len = int(np.round(128/scale_f, 5))
    step_len = 32
#     overlap = int(step_len/2)
#     high_overlap = int(np.round(overlap / 2 * scale_f, 5))
    high_overlap = 192
#    step = step_len - overlap
    step = 16

    BM_G = BatchMaker.\
        BatchMaker(device=device, to_low_idx=to_low_idx, path=G_image_path,
                   sf=scale_f, dims=n_dims, stack=False,
                   down_sample=down_sample, low_res=not down_sample,
                   rot_and_mir=False, squash=squash, super_sample=super_sample)
    im_3d = BM_G.all_image_batch()

    if all_pore_input:
        im_3d[:] = 0
        im_3d[:, 0] = 1

    if input_with_noise:
        input_size = im_3d.size()
        # make noise channel and concatenate it to input:
        noise = torch.randn(input_size[0], 1, *input_size[2:], device=device, dtype=im_3d.dtype)
        im_3d = torch.cat((im_3d, noise), dim=1)

    A = G_net(im_3d[:, :, :32, :32, :32])
    A = A.to_here()
    A = A.detach().cpu()
    A = ImageTools.fractions_to_ohe(A)
    A = ImageTools.one_hot_decoding(A).astype('int8').squeeze()
    imwrite(progress_main_dir + '/' + file_name, A)
