############################################################### IMPORTANT! ##################################################################
# Main credits for developing the algorithm and providing the code goes to the Amir Dahari, Steve Kench, Isaac Squires, and Samuel J. Cooper
# from the Dyson School of Design Engineering, Imperial College London, London SW7 2DB, UK, 
# E-mail: a.dahari@imperial.ac.uk; samuel.cooper@imperial.ac.uk
# Their main repository can be found in https://github.com/tldr-group/SuperRes at https://github.com/tldr-group/SuperRes

# We applied the provided algorithm to the images of rocks. Our modifications to the code are minor and include:
# 1. Increasing the Generator and Discriminator Networks
# 2. Distributing the Generator model into 3 GPUS with the Distributed Data Parallel functionality
# 3. Minor refinements and additional comments regarding the functionality 
#############################################################################################################################################

import LearnTools
import Networks_PIPE
from BatchMaker import *
import os
import time
import random
import torch.nn as nn
import math
import torch.optim as optim
import torch.utils.data
import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.pipeline.sync import Pipe
modes = ['bilinear', 'trilinear']

# Initialize the distributed backend for the Distributed Pipeline Parallel implementation
dist.init_process_group(backend='nccl')
torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

# Exiting the curernt directory to access training datasetand other execution .py files
if os.getcwd().endswith('code_berea_32_new_x8_D128_loss100'):
    os.chdir('..')

# Parsing arguments:
parser = argparse.ArgumentParser()

args = LearnTools.return_args(parser)

progress_dir, wd, wg = args.directory, args.widthD, args.widthG
n_res_blocks, pix_distance = args.n_res_blocks, args.pixel_coefficient_distance
num_epochs, g_update, n_dims = args.num_epochs, args.g_update, args.n_dims
phases_to_low = args.phases_low_res_idx
D_dimensions_to_check, scale_f = args.d_dimensions_to_check, args.scale_factor
rotation, anisotropic = args.with_rotation, args.anisotropic
rotations_bool, down_sample = args.rotations_bool, args.down_sample
super_sampling = args.super_sampling

# Define / create a folder for storing the trained models 
if not os.path.exists(ImageTools.progress_dir + progress_dir):
    os.makedirs(ImageTools.progress_dir + progress_dir)

PATH_G = 'progress/' + progress_dir + '/g_weights.pth'
PATH_D = 'progress/' + progress_dir + '/d_weights.pth'
eta_file = 'eta.npy'

# Root directory for dataset
dataroot = "data/"

# Define the training dataset: D_images = 2D HR LSM segmented images; G_image = 3D LR micro-CT segmented image
D_images = [dataroot + d_path for d_path in args.d_image_path]
G_image = dataroot + args.g_image_path

# G and D slices to choose from
g_batch_slices = [0]  # in 3D different views of the cube, better to keep it as 0

# adding 45 degree angle instead of z axis slices
forty_five_deg = False

# Batch sizes during training
batch_size_G_for_D, batch_size_G, batch_size_D = 2, 32, 128

# Number of GPUs available. In this implementation, we distribute the model into 3 GPUS with the Distributed Pipeline Parallel. Even 
# though for the LR input size of 32^3 it is not requires, the inputs with size 64^3 impossible to run without this modification
ngpu = 3

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print('device is ' + str(device))

# The material indices to low-res. In other words, which groups will be downsampled in SR to compare with the LR image. This line is present
# because the algorithm is capable of generating groups not presented in the LR input, training from the HR examples. In this case, the
# number of groups in LR and SR is different, and to calculate MSE between them, we need to specify which groups to downsample in SR. Doesn't include channel
# for the pore 
to_low_idx = torch.LongTensor(phases_to_low).to(device)

# Number of channels in the training images.
nc_g = 1 + to_low_idx.size()[0] + 1  # 1 channel for pore plus number of material phases to low res plus 1 channel for noise

# number of iterations in each epoch. For some reason, Dahari et al. stated it like this, and we just left it 
epoch_iterations = 10000 // batch_size_G

# Learning rate for optimizers
lr = 0.0001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Learning parameter for gradient penalty
Lambda = 10

# When to save progress
saving_num = 50

def save_differences_and_metrics(input_to_g, output_of_g, down, save_dir, filename, hr_metrics, generator, with_deg=False):
    """
    Saves the image which demonstrates the current model's capabilities to Super-Resolve the input images. 
    The image contains 4 rows and 2 columns. 
    Each column corresponds to a particular example. 
    Row N1 - 2D slice from the 3D LR input. 
    Row N3 - corresponding 2D slice from the 3D SR output. 
    Row N2 - the corresponding 2D slice from the downsampled 3D SR ouput. 
    Row N4 - the random slice from the 3D SR output
    """
    images = [input_to_g.clone().detach().cpu()]
    g_output = output_of_g.cpu()
    down = down.detach().cpu()
    metrics_loss = ImageTools.log_metrics(g_output, hr_metrics)
    if metrics_loss < 0.015:  # mean difference is smaller than 1.5%
        difference_str = str(np.round(metrics_loss, 4))
        torch.save(generator.state_dict(), PATH_G + difference_str)
        wandb.save(PATH_G + difference_str)
    images = images + [down, g_output]
    ImageTools.plot_fake_difference(images, save_dir, filename, with_deg)

if __name__ == '__main__':

    # 1. Start a new run
    print('Initiating the project')
    print(progress_dir)
    
    # Initiate the weight and biases project. It will help to monitor the functionality online
    wandb.init(project='SuperRes', config=args, name=progress_dir)
    print('The project was initiated OK')

    # The batch makers, models, and optimizers for the Discriminator. If the rock is heterogeneous, use 3 different (for xy-, yz-, and xz- directions); if
    # the rock is homogeneous, use the same for all the planes:
    D_BMs, D_nets, D_optimisers = Networks_PIPE.return_D_nets(ngpu, n_dims, device, lr, beta1, anisotropic, D_images, scale_f, rotation, rotations_bool)
    
    # Number of HR phases:
    nc_d = len(D_BMs[0].phases)
    
    # Average volume fraction and surface area matrics calculated in BatchMaker for the whole High-Resolution 2D dataset:
    hr_slice_metrics = D_BMs[0].hr_metrics

    # Define the Batch Maker for the Generator. It will sample sub-volumes from the 3D LR segmented training dataset
    BM_G = BatchMaker(device=device, to_low_idx=to_low_idx, path=G_image, sf=scale_f, dims=n_dims, stack=False, down_sample=down_sample, 
                      low_res=not down_sample, rot_and_mir=False)

    # Create the down-sample object. It is responsible for downsampling the Super-Resolution to compare with Low-Resolution. Two options available: either
    # nearest-neighbor interpolation, either downsampling with the Gaussian convolution. "super-sampling" controlls, which one is implemented
    down_sample_object = LearnTools.DownSample(n_dims, to_low_idx, scale_f, device, super_sampling).to(device)
    
    # Define the Generator model. This one is implemented with the Distributed Pipeline Parallel and 3 GPUs
    model_G = Networks_PIPE.G_PP(device, nc_g, nc_d, BM_G.scale_factor)
    wandb.watch(model_G, log='all')

    # If you don't need Distributed Pipeline Parallel implementation, create your own 3D Generator netwotk, define it in line 163, delete line 199,
    # and run the code below
    # if (device.type == 'cuda') and (ngpu > 1):
    #     model_G = nn.DataParallel(model_G, list(range(ngpu)))

    # Setup Adam optimizers for G
    optimizerG = optim.Adam(model_G.parameters(), lr=lr, betas=(beta1, 0.999))

    def generate_fake_image(detach_output=True, same_seed=False, batch_size=batch_size_G_for_D):
        """
        This function samples sub-volumes of 3D LR segmented micro-CT and concatenate it with noize for training stabilization. Different "detauch_output" is
        used in Generator or Discriminator training
        :param detach_output: to detach the tensor output from gradient memory.
        :param same_seed: generate the same random seed.
        :param batch_size: the batch size of the fake images.
        :return: the generated image from G
        """
        # Generate batch of G's input:
        g_slice = random.choice(g_batch_slices)
        input_to_G = BM_G.random_batch_for_fake(batch_size, g_slice)
        input_size = input_to_G.size()
        # make noise channel and concatenate it to input:
        if same_seed:  # for plotting reasons, have the same noise input each time.
            torch.manual_seed(0)
            noise = torch.randn(input_size[0], 1, *input_size[2:], device=device)
            torch.manual_seed(torch.seed())  # return to randomness
        else:
            noise = torch.randn(input_size[0], 1, *input_size[2:], device=device)
        input_to_G = torch.cat((input_to_G, noise), dim=1)

        # Generate fake image batch with G
        crop = 8
        RREF = model_G(input_to_G)
        BACK = RREF.to_here() # this line is here because of the DPP functionality
        with_edges = BACK
        without_edges = BACK[..., crop:-crop, crop:-crop, crop:-crop] # the image is cropped to avoid artifacts from the Convolutional layers
        if detach_output:
            return input_to_G, with_edges.detach(), without_edges.detach()
        else:
            return input_to_G, with_edges, without_edges


    def take_fake_slices(fake_image, perm_idx):
        """
        This function slices the generated 3D Super-Resolution volume in perm_idx plane
        :param fake_image: The fake image to slice at all directions.
        :param perm_idx: The permutation index for permutation before slicing.
        :return: batch of slices from the 3d image (if 2d image, just returns the image)
        """
        if n_dims == 3:
            perm = perms_3d[perm_idx]
            # permute the fake output of G to make it into a batch
            # of images to feed D (each time different axis)
            fake_slices_for_D = fake_image.permute(0, perm[0], 1, *perm[1:])
            # the new batch size feeding D:
            batch_size_new = batch_size_G_for_D * D_BMs[0].high_l
            # reshaping for the correct size of D's input
            return fake_slices_for_D.reshape(batch_size_new, nc_d, D_BMs[0].high_l, D_BMs[0].high_l)
        else:  # same 2d slices are fed into D
            return fake_image


    ################
    # Training Loop!
    ################

    steps = epoch_iterations
    print("Starting Training Loop...")
    start = time.time()

    for epoch in range(num_epochs):

        j = np.random.randint(steps)  # to see different slices
        for i in range(steps):

            #######################
            # (1) Update D network:
            #######################

            _, _, fake_for_d = generate_fake_image(detach_output=True) # detauch output in training the Discriminator

            for k in range(math.comb(n_dims, 2)): # meaning for each plane: xy-, yz-, xz-
                BM_D, netD, optimizerD = D_BMs[k], D_nets[k], D_optimisers[k]

                # Train with all-real batch
                netD.zero_grad()

                # Batch of real high res for D
                high_res = BM_D.random_batch_for_real(batch_size_D)

                # Forward pass real batch through D
                output_real = netD(high_res).view(-1).mean()

                # obtain fake slices from the fake image
                fake_slices = take_fake_slices(fake_for_d, k)

                # Classify all fake batch with D
                output_fake = netD(fake_slices).view(-1).mean()

                min_batch = min(high_res.size()[0], fake_slices.size()[0])
                
                fake_slices = fake_slices.to(device)
                # Calculate gradient penalty
                gradient_penalty = LearnTools.calc_gradient_penalty(netD, high_res[:min_batch],
                                          fake_slices[:min_batch], batch_size_D, BM_D.high_l, device, Lambda, nc_d)

                # Discriminator is trying to minimize:
                d_cost = output_fake - output_real + gradient_penalty
                
                # Calculate gradients for D in backward pass
                d_cost.backward()
                optimizerD.step()

                # Calculate the wass parameter for tracking
                wass = abs(output_fake.item() - output_real.item())
                
                # we can delete some variables to clear the cash 
#            del _
#            del fake_for_d

            #######################
            # (2) Update G network:
            #######################

            if (i % g_update) == 0: # for stabilization purposes, the Generator may be trained less often than the Discriminator. "g_update" controlls it
                model_G.zero_grad()
                # generate fake again to update G:
                low_res, fake_for_g_vwl, fake_for_g = generate_fake_image(detach_output=False)
                # save the cost of g to add from each axis:
                g_cost = 0
                # go through each plane
                for k in range(math.comb(n_dims, 2)):
                    netD, optimizerD = D_nets[k], D_optimisers[k]
                    fake_slices = take_fake_slices(fake_for_g, k)
                    # perform a forward pass of all-fake batch through D
                    fake_output = netD(fake_slices).view(-1).mean()

                    # track the Discriminator output in the weights and biases
                    if k == 0:
                        wandb.log({'yz_slice': fake_output})
                    if k == 1:
                        wandb.log({'xz_slice': fake_output})

#                    del fake_slices
   
                    # get the voxel-wise-distance loss
                    low_res_without_noise = low_res[:, :-1]  # without noise
                    pix_loss = down_sample_object.voxel_wise_distance(fake_for_g_vwl, low_res_without_noise)

                    # Calculate G's loss based on this output
                    if pix_loss.item() > 0.05:
                        g_cost += -fake_output + pix_distance * pix_loss
                    else: # technically, we never achieved such threshold
                        g_cost += -fake_output

                # Calculate gradients for G
                g_cost.backward()
                # Update G
                optimizerG.step()
                wandb.log({"pixel distance": pix_loss})
                wandb.log({"wass": wass})
                wandb.log({"real": output_real, "fake": output_fake})

            # Output training stats
            if i == j or i == 0:
                ImageTools.calc_and_save_eta(steps, time.time(), start, i, epoch, num_epochs, eta_file)

                with torch.no_grad():  # only for plotting
                    g_input_plot, for_down, g_output_plot = generate_fake_image(detach_output=True, same_seed=True, batch_size=2)

                    downsampled = down_sample_object(for_down)     # was not presented in the initial code

                    # plot input without the noise channel
                    save_differences_and_metrics\
                        (g_input_plot[:, :-1], g_output_plot, downsampled, progress_dir, 'running slices', hr_slice_metrics, model_G, forty_five_deg)
            print(i, j)

        if (epoch % 3) == 0:
            torch.save(model_G.state_dict(), PATH_G)
            wandb.save(PATH_G)
            if (epoch % 60) == 0:
                PATH_G_wo_ext = PATH_G.split('.')[0]
                torch.save(model_G.state_dict(), PATH_G_wo_ext + str(epoch) + '.pth')
                wandb.save(PATH_G_wo_ext + str(epoch) + '.pth')

    print('finished training')