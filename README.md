# Improving prostate cancer diagnostic pathway with a multi-parametric generative self-supervised approach.

Prostate cancer (PCa) is the fifth leading cause of death worldwide. However, diagnosis of PCa based on MRI is challenging because of the time it takes to analyze the images and the variability between the readers. 
Convolutional neural networks (CNNs) have been the de facto standard for nowadays 3D and 2D medical image classification and segmentation. However, CNNs suffer when the amount of data is scarce, being hard to train and not reaching optimal solutions when trained with a limited amount of data. Self-supervised learning techniques aim to train CNNs in such a way that they’re able to learn even in limited data regimes.

In this work, we explored a self-supervised approach to deal with a limited amount of magnetic resonance images (MRI) of the prostate using the [PROSTATEx Challenge](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=23691656) dataset.
We implemented a self-supervised approach able to integrate bi-parametric MRI (ADC and T2w). 
The approach were tested with classification between abnormal (MRI with tumors) and control images (without tumors) in a limited amount of data regime

## Installation

### Prerequisites
   - [PROSTATEx Challenge dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=23691656)
   - [Dependencies](slurm/conda_env.yaml)

### Conda environment setup on UiS AI Lab

   - Clone the repository
   
    $ git clone https://github.com/cgacga/MSc_ProstateCancer

   - Follow the quickstart guide [UiS AI Lab - Slurm](https://github.com/tlinjordet/gpu_ux_uis#environment-setup-script)
     - Replace the bash file with [slurm/env_setup.sh](slurm/env_setup.sh)


## Running the tests

Setting global parameters:
```python
parameters.set_global(
        data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
        self_superviced = True, #if False = Fully supervised approach (No augmentation and training of autoencoder, unless specified otherwise on a MRI sequence level)
        backbone_name = "vgg16", #["vgg16","vgg19"]
        encoder_weights = None, #[None,"imagenet"]
        
        # 3D Image Augmentation
        minmax_shape_reduction  = [15,15], #[min,max] shape of the patch, if min != max a random integer in the interval is chosen.
        minmax_augmentation_percentage  = [60,90], #Percentage of patches to augment across patients
        mask_vs_rotation_percentage = 100, # 0 = only rotation and flipping, 100 = only masking

        # Autoencoder parameters
        autoencoder_epocs = 300,
        autoencoder_batchsize = 2,

        # Classifier training parameters
        classifier_freeze_encoder = True, # True = freezes encoder & bottleneck (Linear probing), False = Fine-tuning
        classifier_multi_dense = False, # Adds top layer structure of vggnet (4096 x2)
        classifier_train_epochs = 30,
        classifier_train_batchsize = 2,
        classifier_train_learning_rate = 1e-5,          

        # Bootstrapping parameters
        bootpercentage = 1 #100% of data used in the evaluation, 0.5 = 50% used
        )
```

Adding MRI sequence:
  All parameters from global can be changed on a MRI sequence level if needed.

```python
parameters.add_modality(
        modality_name = "ADC", # Name of the sequence according to the "Series Description" in the metadata file
        reshape_dim=(32,128,96), # Shape must be divisible by 32
        skip_modality=False, # Added as a standalone sequence, can be skipped if multiple sequences are added
        autoencoder_epocs = 500,
        autoencoder_batchsize = 32,
    )
```

Adding bi-parametric MRI
```python
parameters.add_modality(
        modality_name = "t2tsetra", 
        reshape_dim=None, # Selects the most frequent shape of the sequences
        skip_modality=True # The sequence is not included as a standalone model
    )
    
parameters.join_modalities( 
        ["ADC", "t2tsetra"], # Sequences to include
        decoder_filters = (256, 128, 64, 32, 16),
        encoder_method = "maxpool", # Reshaping method if the sequences does not have equal shape, must be one of either up = ["upsample","transpose","padd"] or down = ["maxpool","avgpool","reshape", "crop"]
        decoder_method = "upsample", # Opposite of the encoder_method
        center_filter = 512, # Filter size in the bottleneck
        encode_try_maxpool_first = True, # Whether to reshape before or after the final maxpooling layer before the bottleneck
        decode_try_upsample_first = True, # Whether to reshape before or after the first upsampling layer after the bottleneck
        merge_method = "avg" # Merging layer ["concat","avg","add","max","multiply"]
        )
```
### Sample

## Built With
   - [Segmentation models 3D Zoo](https://github.com/ZFTurbo/segmentation_models_3D) by [https://github.com/ZFTurbo](ZFTurbo) - Used to create the Unet architecture
