################################################################## IMPORTANT! ##################################################################
Credits for developing the algorithm and providing the code goes to the Amir Dahari, Steve Kench, Isaac Squires, and Samuel J. Cooper from the Dyson School 
of Design Engineering, Imperial College London, London SW7 2DB, UK, E-mail: a.dahari@imperial.ac.uk; samuel.cooper@imperial.ac.uk
Their main repository can be found in https://github.com/tldr-group/SuperRes at https://github.com/tldr-group/SuperRes

Our modifications to the code include:
 1. Increasing the Generator and Discriminator Networks
 2. Distributing the Generator model into 3 GPUS with the Distributed Data Parallel functionality
 3. Minor refinements and additional comments regarding the functionality
################################################################################################################################################

Our main contribution is the adjustment of the 8x Super-Resolution algorithm for the segmented 3D micro-CT images of rocks. 
All details are provided in the attached paper.

In this work, the 3D Generator was trained with the Distributed Pipeline Parallel (DPP) functionality. This way, the 3D Generator model was distributed into
3 A100 GPUs. Such modification allowed us to experiment with inputs of size 64^3 voxels and outputs of size 512^3 (8x times larger). However, in this case,
even with DPP we had to significantly reduce the number of parameters in the Generator model.
Even though in this particular example we finished with inputs of size 32^3 voxels, some types of rocks may require inputs of larger size. 
To use DPP, keep the --DPP as True

If you are limited with the number of available GPUs, or if your don't need DPP for training, keep the --DPP as False. In this case, you may experience the
Out Of Memory (OOM) error. The easiest way to avoid it is either decrease the number of parameters (channels) in the Generator, either reduce the input size
of the 3D LR segmented sub-volume (managed in the BatchMaker). 
However, it is crusual to use the sub-volumes of size large enough to contain at least 2 different grains (and their contact)

To use --DPP as False, your model should fit into single GPU for a particular batch size. 
In this case, if you have several available GPUs, you can run training with Distributed Data Parallel (DDP). To do so, keep --DDP as True

The bottom line is, using the presented approach for different materials in different resolutions requires individual tuning to manage memory requirements
and performance capability.

The training can be launched with the following command

torchrun Architecture_PIPE.py -d 8x_Super-Resolution --with_rotation -phases_idx 1 2 3 -sf 8 -g_image_path Berea_CT_full.tiff -d_image_path Berea_CSLM_clay_gen.tif
