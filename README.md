# Important  
Credits for developing the algorithm and providing the code go to the Amir Dahari, Steve Kench, Isaac Squires, and Samuel J. Cooper from the Dyson School 
of Design Engineering, Imperial College London    

Their publication in Advanced Energy Materials: https://onlinelibrary.wiley.com/doi/epdf/10.1002/aenm.202202407     
Their main repository can be found in https://github.com/tldr-group/SuperRes at https://github.com/tldr-group/SuperRes

# Contribution
Our modifications to the code include:
 1. Increasing Generator and Discriminator Networks
 2. Training on larger sub-volumes due to the nature of rock materials (larger grains require large images)
 3. Distributing the Generator model into 3 GPUs with the Distributed Data Parallel functionality
 4. Minor refinements and additional comments regarding the functionality

Our main contribution is the adjustment of the 8x Super-Resolution algorithm for the segmented 3D micro-CT images of rocks. In this example, we prepared the training dataset and tuned the model for Berea sandstone. Below, we demonstrate the performance of the model. As input, we took the Low-Resolution image, and as output, we produced Super-Resolution image.

![Super-Resolution results for Berea sandstone](GH_image/GH_1.png)  
![Super-Resolution results for Berea sandstone](GH_image/GH_2.png)
![Super-Resolution results for Berea sandstone](GH_image/GH_3.png)

All details are provided in the attached paper.

# Implementation details 
1. In this work, the 3D Generator was trained with the Distributed Pipeline Parallel (DPP) functionality. This way, the 3D Generator model was distributed into three A100 GPUs. Such modification allowed us to experiment with inputs of size 64^3 voxels and outputs of size 512^3 (8x times larger). However, in this case, even with DPP we had to significantly reduce the number of parameters in the Generator model.
Even though in this particular example we finished with inputs of size 32^3 voxels, some types of rocks may require inputs of larger size. 
To use DPP, keep the ```--DPP``` as ```True```.

2. If you are limited with the number of available GPUs, or if your don't need DPP for training, keep the ```--DPP``` as ```False```. In this case, you may experience the Out Of Memory (OOM) error. The easiest way to avoid it is either to decrease the number of parameters (channels) in the Generator, either to reduce the input size of the 3D LR segmented sub-volume (managed in the BatchMaker.py). 
However, it is crusual to use the sub-volumes of size large enough to contain at least two different grains (and their contact).

To use ```--DPP``` as ```False```, your model should fit into single GPU for a particular batch size. 
In this case, if you have several available GPUs, you can run training with Distributed Data Parallel (DDP). To do so, keep ```--DDP ```as ```True```.  

4. As Low-Resolution input, the algorithm uses 3D .tiff files. As High-Resolution ground truth, the algorithm uses a stack of 2D images, also in the .tiff format. Each value in these images represent a particular phase. In this example, we have a similar number of phases in Low-Resolution and High-Resolution images. However, the number of phases in High-Resolution dataset can be more (but not less) than in Low-Resolution dataset. In this case, the algorithm will try to insert the missing phases based on the High-Resolution example  

5. Keep 3D Low-Resolution and 2D High-Resolution datasets in the ![data](data) folder  

6. You should understand that this approach is not a magic wand. Using the presented algorithm for different materials in different resolutions requires individual tuning (sometimes significant) of input size and model size to manage memory requirements and generating performance.  

7. For convenience, you may use the provided environment.   

8. The training can be launched with the following command:

```
torchrun Architecture_PIPE.py -d 8x_Super-Resolution --with_rotation -phases_idx 1 2 3 -sf 8 -g_image_path Berea_CT_full.tiff -d_image_path Berea_CSLM_clay_gen.tif --DPP True --DDP False
```
