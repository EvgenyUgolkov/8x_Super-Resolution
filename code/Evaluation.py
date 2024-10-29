import os
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

# Initialize the distributed backend
dist.init_process_group(backend='nccl')
torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

# Parsing arguments:
parser = argparse.ArgumentParser()

args = LearnTools.return_args(parser)

progress_dir, wd, wg = args.directory, args.widthD, args.widthG
n_res_blocks, pix_distance = args.n_res_blocks, args.pixel_coefficient_distance
num_epochs, g_update, n_dims = args.num_epochs, args.g_update, args.n_dims
squash, down_sample = args.squash_phases, args.down_sample
D_dimensions_to_check, scale_f = args.d_dimensions_to_check, args.scale_factor
size_to_evaluate, separator = args.volume_size_to_evaluate, args.separator
g_file_name, super_sample = args.g_image_path, args.super_sampling
phases_to_low, g_epoch_id = args.phases_low_res_idx, args.g_epoch_id
DPP = args.DPP
nc_d = args.n_phases

# Exiting the curernt directory to access LR data and pre-trained Generator
if os.getcwd().endswith('code'):
    os.chdir('..')

progress_main_dir = 'progress/' + progress_dir
# progress_main_dir = 'progress'
path_to_g_weights = progress_main_dir + '/g_weights' + g_epoch_id + '.pth'
# path_to_g_weights = progress_main_dir + '/g_weights_large_slice.pth'
G_image_path = 'data/' + g_file_name
# G_image_path = 'data/new_vol_down_sample.tif'

rand_id = str(np.random.randint(10000))

file_name = 'generated_tif' + rand_id + '.tif'
crop_to_cube = False
input_with_noise = True
all_pore_input = False

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

# Define the Generator model
if DPP: #This one is implemented with the Distributed Pipeline Parallel and 3 GPUs
    G_net = Networks_PIPE.G_PP(device, nc_g, nc_d, scale_f)
else:
    G_net = Networks_PIPE.Generator3D(nc_g, nc_d, scale_f).to(device)

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

    nz1, nz2, nz3 = size_to_evaluate
    first_img_stack = []
    with torch.no_grad():
        last_ind1 = int(np.ceil((nz1-step_len)/step))
        for i in range(last_ind1 + 1):
            print('large step = ' + str(i))
            if i == last_ind1:
                first_lr_vec = im_3d[..., nz1-step_len:nz1, :, :]
            else:
                first_lr_vec = im_3d[..., i*step:i*step+step_len, :, :]
            second_img_stack = []
            last_ind2 = int(np.ceil((nz2-step_len)/step))
            for j in range(last_ind2 + 1):
                print('middle step = ' + str(j))
                if j == last_ind2:
                    second_lr_vec = first_lr_vec[..., :, nz2-step_len:nz2, :]
                else:
                    second_lr_vec = first_lr_vec[..., :, j * step:j * step + step_len, :]
                third_img_stack = []
                last_ind3 = int(np.ceil((nz3-step_len)/step))
                for k in range(last_ind3 + 1):
                    print('small step = ' + str(k))
                    if k == last_ind3:
                        third_lr_vec = second_lr_vec[..., :, :, nz3-step_len:nz3]
                    else:
                        third_lr_vec = second_lr_vec[..., :, :, k * step:k * step + step_len]
                    g_output = G_net(third_lr_vec)
                    if DPP:
                        g_output = g_output.to_here()
                    g_output = g_output.detach().cpu()
                    g_output = ImageTools.fractions_to_ohe(g_output)
                    g_output_grey = ImageTools.one_hot_decoding(g_output).astype('int8').squeeze()
                    if k == 0:  # keep the beginning
                        g_output_grey = g_output_grey[:, :, :high_overlap]
                    elif k == last_ind3:  # keep the middle+end
                        g_output_grey = g_output_grey[:, :, -high_overlap:]
                    else:  # keep the middle
                        g_output_grey = g_output_grey[:, :, -high_overlap:high_overlap]
                    third_img_stack.append(np.int8(g_output_grey))
                res2 = np.concatenate(third_img_stack, axis=2)
                if j == 0:
                    res2 = res2[:, :high_overlap, :]
                elif j == last_ind2:
                    res2 = res2[:, -high_overlap:, :]
                else:
                    res2 = res2[:, -high_overlap:high_overlap, :]
                second_img_stack.append(res2)
            res1 = np.concatenate(second_img_stack, axis=1)
            if i == 0:
                res1 = res1[:high_overlap, :, :]
            elif i == last_ind1:
                res1 = res1[-high_overlap:, :, :]
            else:
                res1 = res1[-high_overlap:high_overlap, :, :]
            first_img_stack.append(res1)
    img = np.concatenate(first_img_stack, axis=0)
    img = img[crop:-crop, crop:-crop, crop:-crop]
    low_res = np.squeeze(ImageTools.one_hot_decoding(im_3d.cpu()))
    if all_pore_input:
        imwrite(progress_main_dir + '/' + file_name + '_pore', img)
    else:
        imwrite(progress_main_dir + '/' + file_name, img)

    # also save the low-res input.
    imwrite(progress_main_dir + '/' + file_name.split('.')[0] + '_low_res.tif',
           low_res)
