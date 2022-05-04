import os, sys, itertools
from parameters import parameters

def task_parameters():
    task_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
    task_max = int(os.environ['SLURM_ARRAY_TASK_MAX'])

    epochs = 100
    encoder_weights = "imagenet"
    encoder_freeze = True
    no_adjacent = False
    
    backbone_name = ["vgg16", "resnet18"]
    activation = ["sigmoid", "softmax"]
    decoder_block_type = ["upsampling", "transpose"]
    #learning_rate = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    learning_rate = [1e-2, 1e-4, 1e-6]

    #encoder_freeze = [True, False]

    minmax_augmentation_percentage  = [15,16]#[[15,15], [10,15]]
    minmax_shape_reduction  = [6,7]#[[10,10], [5,15]]
    mask_vs_rotation_percentage = 100#[0,50]

    iterate = list(itertools.product(
                            backbone_name,
                            activation,
                            decoder_block_type,
                            learning_rate
                            ))#,
                            #minmax_augmentation_percentage,
                            #minmax_shape_reduction,
                            #mask_vs_rotation_percentage))



    if task_idx > len(iterate):
        sys.exit("SLURM_ARRAY_TASK_MAX larger than number of tasks")

    print("Task:", task_idx, "/", len(iterate))

    backbone_name = iterate[task_idx][0]
    activation = iterate[task_idx][1]
    decoder_block_type = iterate[task_idx][2]
    learning_rate = iterate[task_idx][3]
    #encoder_freeze = iterate[task_idx][4]
    #minmax_augmentation_percentage = iterate[task_idx][4]
    #minmax_shape_reduction = iterate[task_idx][5]
    #mask_vs_rotation_percentage = iterate[task_idx][6]


    job_name = f"{backbone_name}_{activation}_{decoder_block_type}_lr{learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_{task_idx}"
    #_{encoder_weights}_{encoder_freeze}

    parameters.set_global(
        data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
        job_name = job_name,
        backbone_name = backbone_name,
        activation = activation,
        encoder_weights = encoder_weights,
        encoder_freeze = encoder_freeze,
        decoder_block_type = decoder_block_type, 
        epochs = epochs,
        learning_rate  = learning_rate,
        minmax_shape_reduction  = minmax_shape_reduction,
        minmax_augmentation_percentage  = minmax_augmentation_percentage,
        mask_vs_rotation_percentage = mask_vs_rotation_percentage,
        no_adjacent = no_adjacent
        )
    parameters.add_modality(
        modality_name = "ADC", 
        batch_size=32, 
        reshape_dim=(32,128,96))
    parameters.add_modality(
        modality_name = "t2tsetra", 
        reshape_dim=None,  
        batch_size=2)


