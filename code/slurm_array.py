import os, sys, itertools
from params import *
import tensorflow as tf
from merged_model import get_merged_model


def single_run(index):

    backbone_name = "vgg16"
    encoder_weights = "imagenet"
    encoder_freeze = False
    activation = "sigmoid"
    decoder_block_type = "upsampling"
    encoder_freeze = False

    modality_name = ["ADC","t2tsetra"]       
    learning_rate = [1e-3, 1e-4]
    cube = [[[15,15],[60,60],100],[[10,20],[40,60],50]]


    iterate = list(itertools.product(
                            cube,
                            learning_rate,
                            modality_name
                            ))

    cube_params, learning_rate, modality_name = iterate[index]
    minmax_shape_reduction, minmax_augmentation_percentage,mask_vs_rotation_percentage = cube_params

    if modality_name == "ADC":
        epochs = 500
        batch_size = 32
        reshape_dim = (32,128,96)
    elif modality_name == "t2tsetra":
        epochs = 250
        batch_size = 2
        reshape_dim = None

    job_name = f"e{epochs}_lr{learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_bs{batch_size}"

    parameters.set_global(
            data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
            job_name = job_name,
            encoder_freeze = encoder_freeze,
            decoder_block_type = decoder_block_type, 
            epochs = epochs,
            learning_rate  = learning_rate,
            minmax_shape_reduction  = minmax_shape_reduction,
            minmax_augmentation_percentage  = minmax_augmentation_percentage,
            mask_vs_rotation_percentage = mask_vs_rotation_percentage
            )

    parameters.add_modality(
        modality_name = modality_name, 
        reshape_dim=reshape_dim,  
        batch_size=batch_size)

    try:
        task_max = int(os.environ['SLURM_ARRAY_TASK_MAX'])
        print(f"Task: {index}/{len(iterate)-1} \t(Task_Max: {task_max})")
    except:
        print(f"Task: {index}/{len(iterate)-1}")

    
    #parameters.set_current(modality_name)

    ####model = get_merged_model()
    #model = sm.Unet(modality.backbone_name, dim osv osv osv)


    # if not os.path.exists(modality.model_path):
    #     os.makedirs(modality.model_path)

    # tf.keras.utils.plot_model(
    # model,
    # show_shapes=True,
    # show_layer_activations = True,
    # expand_nested=True,
    # to_file=os.path.abspath(modality.model_path+f"autoencoder.png")
    # )

    #tf.keras.backend.clear_session()


def merged_run(index):

    epochs = 350
    batch_size = 1
    encoder_freeze = False
    decoder_block_type = "upsampling"

    learning_rate = [1e-3, 1e-4]
    cube = [[[15,15],[60,60],100],[[10,20],[40,60],50]]

    updowm = [["maxpool","upsample",False,True],["upsample","maxpool",False,True]]#,["maxpool","upsample",True,True]]

    # rm /e350_lr0.001_sr15-15_ap60-60_mvsrp100_efFalse_bs1_emupsample_dmmaxpool_cf1024_df1024-64_etmfFalse_dtufTrue/Merged_ADC-t2tsetra/
    # rm /e350_lr0.001_sr15-15_ap60-60_mvsrp100_efFalse_bs1_emmaxpool_dmupsample_cf1024_df1024-64_etmfFalse_dtufTrue/Merged_ADC-t2tsetra/
    # test 128 istedenfor 1024
    filters = [[512,(256, 128, 64, 32, 16)],[1024,(512, 256, 128, 64, 32)],[256,(256, 128, 64, 32, 16)]]#,[512,(512,256, 128, 64, 32)]

    iterate = list(itertools.product(
                            cube,
                            filters,                        
                            learning_rate,
                            updowm
                            ))

    # for i in range(len(iterate)):
    #     print(iterate[i])

    cube_params,filters,learning_rate, updowm = iterate[index]
    center_filter,decoder_filters = filters
    minmax_shape_reduction, minmax_augmentation_percentage,mask_vs_rotation_percentage = cube_params
    encoder_method, decoder_method, encode_try_maxpool_first, decode_try_upsample_first = updowm

        
    job_name = f"e{epochs}_lr{learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_bs{batch_size}_em{encoder_method}_dm{decoder_method}_cf{center_filter}_df{decoder_filters[0]}-{decoder_filters[-1]}_etmf{encode_try_maxpool_first}_dtuf{decode_try_upsample_first}"

    parameters.set_global(
                data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
                job_name = job_name,
                encoder_freeze = encoder_freeze,
                decoder_block_type = decoder_block_type, 
                epochs = epochs,
                learning_rate  = learning_rate,
                minmax_shape_reduction  = minmax_shape_reduction,
                minmax_augmentation_percentage  = minmax_augmentation_percentage,
                mask_vs_rotation_percentage = mask_vs_rotation_percentage,
                batch_size = batch_size
                )


    parameters.add_modality(
        modality_name = "ADC", 
        reshape_dim=(32,128,96),
        skip_modality=True
        )
    parameters.add_modality(
        modality_name = "t2tsetra", 
        reshape_dim=(32,384,384),
        skip_modality=True
        )
    parameters.join_modalities(["ADC", "t2tsetra"], encoder_method = encoder_method, decoder_method=decoder_method, center_filter=center_filter, decoder_filters=decoder_filters, decode_try_upsample_first=decode_try_upsample_first,encode_try_maxpool_first=encode_try_maxpool_first)

    
    try:
        task_max = int(os.environ['SLURM_ARRAY_TASK_MAX'])
        print(f"Task: {index}/{len(iterate)-1} \t(Task_Max: {task_max})")
    except:
        print(f"Task: {index}/{len(iterate)-1}")


    parameters.set_current("Merged")
    
    model = get_merged_model()

    if not os.path.exists(modality.model_path):
        os.makedirs(modality.model_path)

    tf.keras.utils.plot_model(
    model,
    show_shapes=True,
    show_layer_activations = True,
    expand_nested=True,
    to_file=os.path.abspath(modality.model_path+f"autoencoder.png")
    )

    tf.keras.backend.clear_session()




def samesize_run(index):
    
    backbone_name = "vgg16"
    encoder_weights = "imagenet"
    encoder_freeze = False
    activation = "sigmoid"
    decoder_block_type = "upsampling"

    modality_name = ["ADC","t2tsetra"]       
    learning_rate = [1e-3, 1e-4]
    cube = [[[15,15],[60,60],100],[[10,20],[40,60],50]]


    iterate = list(itertools.product(
                            cube,
                            learning_rate,
                            modality_name
                            ))

    cube_params, learning_rate, modality_name = iterate[index]
    minmax_shape_reduction, minmax_augmentation_percentage,mask_vs_rotation_percentage = cube_params

    
    if index == 0:
        batch_size = 1
        reshape_dim = (32,384,384)
    elif index == 1:
        batch_size = 16
        reshape_dim = (32,128,96)
    else:
        raise ValueError("Index out of range")
    
    epochs = 500
    
        

    job_name = f"e{epochs}_lr{learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_bs{batch_size}_SameSize"

    parameters.set_global(
            data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
            job_name = job_name,
            encoder_freeze = encoder_freeze,
            decoder_block_type = decoder_block_type, 
            epochs = epochs,
            learning_rate  = learning_rate,
            minmax_shape_reduction  = minmax_shape_reduction,
            minmax_augmentation_percentage  = minmax_augmentation_percentage,
            mask_vs_rotation_percentage = mask_vs_rotation_percentage,
            batch_size = batch_size
            )

    parameters.add_modality(
        modality_name = "ADC", 
        reshape_dim=reshape_dim,
        skip_modality = True)

    parameters.add_modality(
        modality_name = "t2tsetra", 
        reshape_dim=reshape_dim,
        skip_modality = True)

    # parameters.join_modalities(["ADC", "t2tsetra"], encoder_method = encoder_method, decoder_method=decoder_method, center_filter=center_filter, decoder_filters=decoder_filters, decode_try_upsample_first=decode_try_upsample_first,encode_try_maxpool_first=encode_try_maxpool_first)
    
    center_filter = 512
    decoder_filters = (512,256,128,64,32)

    parameters.join_modalities(["ADC", "t2tsetra"], center_filter=center_filter, decoder_filters=decoder_filters)


    try:
        task_max = int(os.environ['SLURM_ARRAY_TASK_MAX'])
        print(f"Task: {index}/{len(iterate)-1} \t(Task_Max: {task_max})")
    except:
        print(f"Task: {index}/{len(iterate)-1}")

    parameters.set_current("Merged")
    
    model = get_merged_model()


    if not os.path.exists(modality.model_path):
        os.makedirs(modality.model_path)


    tf.keras.utils.plot_model(
    model,
    show_shapes=True,
    show_layer_activations = True,
    expand_nested=True,
    to_file=os.path.abspath(modality.model_path+f"autoencoder.png")
    )

    tf.keras.backend.clear_session()
# def single_run(index):

#     backbone_name = "vgg16"
#     encoder_weights = "imagenet"
#     encoder_freeze = False
#     activation = "sigmoid"
#     decoder_block_type = "upsampling"

#     modality_name = ["ADC","t2tsetra"]       
#     learning_rate = [1e-3, 1e-4]
#     cube = [[[15,15],[60,60],100],[[10,20],[40,60],50]]


#     iterate = list(itertools.product(
#                             cube,
#                             learning_rate,
#                             modality_name
#                             ))

#     cube_params, learning_rate, modality_name = iterate[index]
#     minmax_shape_reduction, minmax_augmentation_percentage,mask_vs_rotation_percentage = cube_params

#     if modality_name == "ADC":
#         epochs = 500
#         batch_size = 32
#         reshape_dim = (32,128,96)
#     elif modality_name == "t2tsetra":
#         epochs = 250
#         batch_size = 2
#         reshape_dim = None

#     job_name = f"e{epochs}_lr{learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_bs{batch_size}"

#     parameters.set_global(
#             data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
#             job_name = job_name,
#             backbone_name = backbone_name,
#             activation = activation,
#             encoder_weights = encoder_weights,
#             encoder_freeze = encoder_freeze,
#             decoder_block_type = decoder_block_type, 
#             epochs = epochs,
#             learning_rate  = learning_rate,
#             minmax_shape_reduction  = minmax_shape_reduction,
#             minmax_augmentation_percentage  = minmax_augmentation_percentage,
#             mask_vs_rotation_percentage = mask_vs_rotation_percentage
#             )

#     parameters.add_modality(
#         modality_name = modality_name, 
#         reshape_dim=reshape_dim,  
#         batch_size=batch_size)

#     try:
#         task_max = int(os.environ['SLURM_ARRAY_TASK_MAX'])
#         print(f"Task: {index}/{len(iterate)-1} \t(Task_Max: {task_max})")
#     except:
#         print(f"Task: {index}/{len(iterate)-1}")


# def merged_run(index):

#     epochs = 350
#     batch_size = 1

#     learning_rate = [1e-3, 1e-4]
#     cube = [[[15,15],[60,60],100],[[10,20],[40,60],50]]

#     updowm = [["maxpool","upsample",False,True],["upsample","maxpool",False,True]]#,["maxpool","upsample",True,True]]

#     filters = [[512,(256, 128, 64, 32, 16)],[1024,(1024, 512, 256, 128, 64)],[256,(256, 128, 64, 32, 16)]]#,[512,(512,256, 128, 64, 32)]

#     iterate = list(itertools.product(
#                             cube,
#                             filters,                        
#                             learning_rate,
#                             updowm
#                             ))

#     # for i in range(len(iterate)):
#     #     print(iterate[i])

#     cube_params,filters,learning_rate, updowm = iterate[index]
#     center_filter,decoder_filters = filters
#     minmax_shape_reduction, minmax_augmentation_percentage,mask_vs_rotation_percentage = cube_params
#     encoder_method, decoder_method, encode_try_maxpool_first, decode_try_upsample_first = updowm

        
#     job_name = f"e{epochs}_lr{learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_bs{batch_size}_em{encoder_method}_dm{decoder_method}_cf{center_filter}_df{decoder_filters[0]}-{decoder_filters[-1]}_etmf{encode_try_maxpool_first}_dtuf{decode_try_upsample_first}"

#     parameters.set_global(
#                 data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
#                 job_name = job_name,
#                 backbone_name = backbone_name,
#                 activation = activation,
#                 encoder_weights = encoder_weights,
#                 encoder_freeze = encoder_freeze,
#                 decoder_block_type = decoder_block_type, 
#                 epochs = epochs,
#                 learning_rate  = learning_rate,
#                 minmax_shape_reduction  = minmax_shape_reduction,
#                 minmax_augmentation_percentage  = minmax_augmentation_percentage,
#                 mask_vs_rotation_percentage = mask_vs_rotation_percentage
#                 )


#     parameters.add_modality(
#         modality_name = "ADC", 
#         reshape_dim=(32,128,96),
#         skip_modality=True
#         )
#     parameters.add_modality(
#         modality_name = "t2tsetra", 
#         reshape_dim=(32,384,384),
#         skip_modality=True
#         )
#     parameters.join_modalities(["ADC", "t2tsetra"], encoder_method = encoder_method, decoder_method=decoder_method, center_filter=center_filter, decoder_filters=decoder_filters, decode_try_upsample_first=decode_try_upsample_first,encode_try_maxpool_first=encode_try_maxpool_first)

    
#     parameters.set_current("Merged")
#     get_merged_model()
#     tf.keras.backend.clear_session()

#     try:
#         task_max = int(os.environ['SLURM_ARRAY_TASK_MAX'])
#         print(f"Task: {index}/{len(iterate)-1} \t(Task_Max: {task_max})")
#     except:
#         print(f"Task: {index}/{len(iterate)-1}")




# def samesize_run(index):
    
#     backbone_name = "vgg16"
#     encoder_weights = "imagenet"
#     encoder_freeze = False
#     activation = "sigmoid"
#     decoder_block_type = "upsampling"

#     modality_name = ["ADC","t2tsetra"]       
#     learning_rate = [1e-3, 1e-4]
#     cube = [[[15,15],[60,60],100],[[10,20],[40,60],50]]


#     iterate = list(itertools.product(
#                             cube,
#                             learning_rate,
#                             modality_name
#                             ))

#     cube_params, learning_rate, modality_name = iterate[index]
#     minmax_shape_reduction, minmax_augmentation_percentage,mask_vs_rotation_percentage = cube_params

    
#     if index == 0:
#         batch_size = 1
#         reshape_dim = (32,384,384)
#     elif index == 1:
#         batch_size = 16
#         reshape_dim = (32,128,96)
#     else:
#         raise ValueError("Index out of range")
    
#     epochs = 500
    
        

#     job_name = f"e{epochs}_lr{learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_bs{batch_size}"

#     parameters.set_global(
#             data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
#             job_name = job_name,
#             backbone_name = backbone_name,
#             activation = activation,
#             encoder_weights = encoder_weights,
#             encoder_freeze = encoder_freeze,
#             decoder_block_type = decoder_block_type, 
#             epochs = epochs,
#             learning_rate  = learning_rate,
#             minmax_shape_reduction  = minmax_shape_reduction,
#             minmax_augmentation_percentage  = minmax_augmentation_percentage,
#             mask_vs_rotation_percentage = mask_vs_rotation_percentage,
#             batch_size = batch_size
#             )

#     parameters.add_modality(
#         modality_name = "ADC", 
#         reshape_dim=reshape_dim,
#         skip_modality = True)

#     parameters.add_modality(
#         modality_name = "t2tsetra", 
#         reshape_dim=reshape_dim,
#         skip_modality = True)

#     # parameters.join_modalities(["ADC", "t2tsetra"], encoder_method = encoder_method, decoder_method=decoder_method, center_filter=center_filter, decoder_filters=decoder_filters, decode_try_upsample_first=decode_try_upsample_first,encode_try_maxpool_first=encode_try_maxpool_first)
    
#     center_filter = 512
#     decoder_filters = (512,256,128,64,32)

#     parameters.join_modalities(["ADC", "t2tsetra"], center_filter=center_filter, decoder_filters=decoder_filters)


#     try:
#         task_max = int(os.environ['SLURM_ARRAY_TASK_MAX'])
#         print(f"Task: {index}/{len(iterate)-1} \t(Task_Max: {task_max})")
#     except:
#         print(f"Task: {index}/{len(iterate)-1}")





# def task_parameters():
#     task_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
#     task_max = int(os.environ['SLURM_ARRAY_TASK_MAX'])

#     epochs = 500
#     encoder_weights = "imagenet"
#     encoder_freeze = False
#     no_adjacent = False
    
    
#     modality_name = ["t2tsetra","ADC"]
#     backbone_name = "vgg16"#["vgg16", "resnet18"]
#     activation = "sigmoid"#["sigmoid", "softmax"]
#     decoder_block_type = "upsampling"#["upsampling", "transpose"]
#     learning_rate = [1e-3, 1e-4, 1e-5, 1e-6]
#     #learning_rate = [1e-3,1e-4,1e-5]

#     #encoder_freeze = [True, False]

#     minmax_augmentation_percentage  = [[60,60], [40,75]]#[15,16]
#     minmax_shape_reduction  = [[15,15], [5,25]]#[6,7]
#     mask_vs_rotation_percentage = [100,50]

#     iterate = list(itertools.product(
#                             #backbone_name,
#                             #activation,
#                             #decoder_block_type,
                            
                            
#                             minmax_augmentation_percentage,
#                             minmax_shape_reduction,
#                             mask_vs_rotation_percentage,
#                             learning_rate,
#                             modality_name
#                             ))



#     #if task_max < (len(iterate)-1):
#     #    sys.exit("SLURM_ARRAY_TASK_MAX smaller than number of tasks")

#     print(f"Task: {task_idx}/{len(iterate)-1} \t(Task_Max: {task_max})")


#     minmax_augmentation_percentage, minmax_shape_reduction, mask_vs_rotation_percentage, learning_rate, modality_name = iterate[task_idx]
#     #backbone_name = iterate[task_idx][0]
#     #activation = iterate[task_idx][1]
#     #decoder_block_type = iterate[task_idx][2]
#     #encoder_freeze = iterate[task_idx][4]
#     #minmax_augmentation_percentage = iterate[task_idx][1]
#     #minmax_shape_reduction = iterate[task_idx][2]
#     #mask_vs_rotation_percentage = iterate[task_idx][6]
#     #bs = [4,8]#[4,8,16,32,64]
#     #batch_size = bs[task_idx]
   


#     job_name = f"{backbone_name}_{activation}_{decoder_block_type}_e{epochs}_lr{learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}"
#     #_{encoder_weights}_

#     parameters.set_global(
#         data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
#         job_name = job_name,
#         backbone_name = backbone_name,
#         activation = activation,
#         encoder_weights = encoder_weights,
#         encoder_freeze = encoder_freeze,
#         decoder_block_type = decoder_block_type, 
#         epochs = epochs,
#         learning_rate  = learning_rate,
#         minmax_shape_reduction  = minmax_shape_reduction,
#         minmax_augmentation_percentage  = minmax_augmentation_percentage,
#         mask_vs_rotation_percentage = mask_vs_rotation_percentage,
#         no_adjacent = no_adjacent
#         )

#     if modality_name == "ADC":
#       batch_size = 32
#       reshape_dim = (32,128,96)
#     elif modality_name == "t2tsetra":
#       batch_size = 2
#       reshape_dim = None

#     parameters.add_modality(
#         modality_name = modality_name, 
#         reshape_dim=reshape_dim,  
#         batch_size=batch_size)

#     # parameters.add_modality(
#     #     modality_name = "ADC", 
#     #     batch_size=32, 
#     #     reshape_dim=(32,128,96))

#     # parameters.add_modality(
#     #     modality_name = "t2tsetra", 
#     #     reshape_dim=None,  
#     #     batch_size=2)


