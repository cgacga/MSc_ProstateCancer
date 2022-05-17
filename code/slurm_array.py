import os, sys, itertools
from params import parameters

def task_parameters():
    task_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
    task_max = int(os.environ['SLURM_ARRAY_TASK_MAX'])

    epochs = 500
    encoder_weights = "imagenet"
    encoder_freeze = True
    no_adjacent = False
    
    
    modality_name = ["t2tsetra","ADC"]
    backbone_name = "vgg16"#["vgg16", "resnet18"]
    activation = "sigmoid"#["sigmoid", "softmax"]
    decoder_block_type = "upsampling"#["upsampling", "transpose"]
    learning_rate = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    #learning_rate = [1e-3,1e-4,1e-5]

    #encoder_freeze = [True, False]

    minmax_augmentation_percentage  = [[60,60], [40,75]]#[15,16]
    minmax_shape_reduction  = [[15,15], [5,25]]#[6,7]
    mask_vs_rotation_percentage = [100,50]

    iterate = list(itertools.product(
                            #backbone_name,
                            #activation,
                            #decoder_block_type,
                            
                            
                            minmax_augmentation_percentage,
                            minmax_shape_reduction,
                            mask_vs_rotation_percentage,
                            learning_rate,
                            modality_name
                            ))



    #if task_max < (len(iterate)-1):
    #    sys.exit("SLURM_ARRAY_TASK_MAX smaller than number of tasks")

    print(f"Task: {task_idx}/{len(iterate)-1} \t(Task_Max: {task_max})")


    minmax_augmentation_percentage, minmax_shape_reduction, mask_vs_rotation_percentage, learning_rate, modality_name = iterate[task_idx]
    #backbone_name = iterate[task_idx][0]
    #activation = iterate[task_idx][1]
    #decoder_block_type = iterate[task_idx][2]
    #encoder_freeze = iterate[task_idx][4]
    #minmax_augmentation_percentage = iterate[task_idx][1]
    #minmax_shape_reduction = iterate[task_idx][2]
    #mask_vs_rotation_percentage = iterate[task_idx][6]
    #bs = [4,8]#[4,8,16,32,64]
    #batch_size = bs[task_idx]
   


    job_name = f"{backbone_name}_{activation}_{decoder_block_type}_e{epochs}_lr{learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}"
    #_{encoder_weights}_

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

    if modality_name == "ADC":
      batch_size = 32
      reshape_dim = (32,128,96)
    elif modality_name == "t2tsetra":
      batch_size = 2
      reshape_dim = None

    parameters.add_modality(
        modality_name = modality_name, 
        reshape_dim=reshape_dim,  
        batch_size=batch_size)

    # parameters.add_modality(
    #     modality_name = "ADC", 
    #     batch_size=32, 
    #     reshape_dim=(32,128,96))

    # parameters.add_modality(
    #     modality_name = "t2tsetra", 
    #     reshape_dim=None,  
    #     batch_size=2)


