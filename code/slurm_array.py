import os, sys, itertools
from params import *
import tensorflow as tf
from merged_model import get_merged_model


def all(index):

    encoder_weights = None #[None,"imagenet"]
    self_superviced = False #[True,False]
    bootpercentage = 1 #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    cube = [[[15,15],[60,60],100],[[10,20],[60,90],50]]
    classifier_freeze_encoder = False #[False,True]
    classifier_multi_dense = False #[False,True]
    batchnorm = True # [False,True]
    dropout = None #0.1, 0.3

    #merged only
    center_filter = 512
    decoder_filters = (256, 128, 64, 32, 16)
    encoder_method, decoder_method, encode_try_maxpool_first, decode_try_upsample_first = ["upsample","maxpool",False,True]
    #test *,*,True,*whatever]

    merge_method = "add"#["concat","avg","add","max","multiply"]
    
    # iterate = list(itertools.product(
    #                         cube
    #                         ))

    # cube_params, = iterate[index]
    minmax_shape_reduction, minmax_augmentation_percentage,mask_vs_rotation_percentage = cube[0]#cube_params


    parameters.set_global(
            data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
            encoder_weights = encoder_weights,
            self_superviced = self_superviced,

            autoencoder_epocs = 5,
            autoencoder_batchsize = 1,
            classifier_train_epochs = 35,
            classifier_train_batchsize = 2, #TEST MED 2 +++
            classifier_test_batchsize = 1,  #TEST MED 2 +++

            batchnorm = batchnorm,
            dropout = dropout,

            bootpercentage = bootpercentage,
            classifier_freeze_encoder = classifier_freeze_encoder,
            classifier_multi_dense = classifier_multi_dense,

            minmax_shape_reduction  = minmax_shape_reduction,
            minmax_augmentation_percentage  = minmax_augmentation_percentage,
            mask_vs_rotation_percentage = mask_vs_rotation_percentage
            
            
            )


    parameters.add_modality(
        modality_name = "ADC", 
        reshape_dim=(32,128,96),  
        autoencoder_batchsize = 32,
        autoencoder_epocs = 35,
        skip_modality=True, 
        classifier_train_batchsize = 32,
        classifier_train_epochs = 2,
        classifier_test_batchsize = 1
        )
    parameters.add_modality(
        modality_name = "t2tsetra", 
        reshape_dim=None,  
        autoencoder_batchsize = 2,
        autoencoder_epocs = 5,
        skip_modality=True, 
        classifier_train_batchsize = 2,
        classifier_train_epochs = 2,
        classifier_test_batchsize = 2
        )
        

    parameters.join_modalities(["ADC", "t2tsetra"],
            decoder_filters = decoder_filters,
            encoder_method = encoder_method,
            decoder_method = decoder_method,
            center_filter = center_filter, 
            decode_try_upsample_first = decode_try_upsample_first, 
            encode_try_maxpool_first = encode_try_maxpool_first,
            merge_method = merge_method)


def single_run(index):

    
    self_superviced = True#[True,False]
    #encoder
    encoder_weights = [None,"imagenet"]
    cube = [[[15,15],[60,60],100],[[10,20],[60,90],50]]

    #test
    classifier_freeze_encoder = [False]#[False,True] #IF TIME
    classifier_multi_dense = [False]#[False,True]    #IF TIME
    
    #eval
    bootpercentage = [1,0.1,0.2,0.5]
    #16

    iterate = list(itertools.product(
                            bootpercentage,
                            classifier_freeze_encoder,
                            classifier_multi_dense,
                            cube,
                            encoder_weights
                            ))

    bootpercentage,classifier_freeze_encoder,classifier_multi_dense,cube_params,encoder_weights = iterate[index]

    minmax_shape_reduction, minmax_augmentation_percentage,mask_vs_rotation_percentage = cube_params


    parameters.set_global(
            data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
            encoder_weights = encoder_weights,
            self_superviced = self_superviced,

            bootpercentage = bootpercentage,
            classifier_freeze_encoder = classifier_freeze_encoder,
            classifier_multi_dense = classifier_multi_dense,

            minmax_shape_reduction  = minmax_shape_reduction,
            minmax_augmentation_percentage  = minmax_augmentation_percentage,
            mask_vs_rotation_percentage = mask_vs_rotation_percentage
            
            )


    parameters.add_modality(
        modality_name = "ADC", 
        reshape_dim=(32,128,96),  
        autoencoder_batchsize = 32,
        autoencoder_epocs = 70,
        skip_modality=False, 
        classifier_train_batchsize = 2,
        classifier_train_epochs = 35,
        classifier_test_batchsize = 2
        )
    parameters.add_modality(
        modality_name = "t2tsetra", 
        reshape_dim=None,  
        autoencoder_batchsize = 2,
        autoencoder_epocs = 70,
        skip_modality=False, 
        classifier_train_batchsize = 2,
        classifier_train_epochs = 35,
        classifier_test_batchsize = 2
        )
        


    try:
        task_max = int(os.environ['SLURM_ARRAY_TASK_MAX'])
        print(f"Task: {index}/{len(iterate)-1} \t(Task_Max: {task_max})")
    except:
        print(f"Task: {index}/{len(iterate)-1}")



def merged_run(index):


    self_superviced = True#[True,False]
    encoder_weights = [None,"imagenet"]
    cube = [[[15,15],[60,60],100],[[10,20],[60,90],50]]
    center_filter = 512
    decoder_filters = (256, 128, 64, 32, 16)
    updown = [["upsample","maxpool",False,True],["maxpool","upsample",True,True]]
    merge_method = ["concat","avg","add","max","multiply"]

    classifier_freeze_encoder = [False]#[False,True]
    classifier_multi_dense = [False]#[False,True]
    
    bootpercentage = [1,0.1,0.2,0.5]

    #160
    

    iterate = list(itertools.product(
                        bootpercentage,
                        classifier_freeze_encoder,
                        classifier_multi_dense,
                        encoder_weights,
                        cube,
                        merge_method,
                        updown
                        ))

    bootpercentage,classifier_freeze_encoder,classifier_multi_dense,encoder_weights,cube_params,merge_method, updown = iterate[index]

    minmax_shape_reduction, minmax_augmentation_percentage,mask_vs_rotation_percentage = cube_params
    encoder_method, decoder_method, encode_try_maxpool_first, decode_try_upsample_first = updown


    parameters.set_global(
            data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
            encoder_weights = encoder_weights,
            self_superviced = self_superviced,

            autoencoder_epocs = 70,
            autoencoder_batchsize = 1,
            classifier_train_epochs = 35,
            classifier_train_batchsize = 2,
            classifier_test_batchsize = 2, 

            bootpercentage = bootpercentage,
            classifier_freeze_encoder = classifier_freeze_encoder,
            classifier_multi_dense = classifier_multi_dense,

            minmax_shape_reduction  = minmax_shape_reduction,
            minmax_augmentation_percentage  = minmax_augmentation_percentage,
            mask_vs_rotation_percentage = mask_vs_rotation_percentage
            )


    parameters.add_modality(
        modality_name = "ADC", 
        reshape_dim=(32,128,96),
        skip_modality=True
        )
    parameters.add_modality(
        modality_name = "t2tsetra", 
        reshape_dim=None,
        skip_modality=True
        )
        

    parameters.join_modalities(["ADC", "t2tsetra"],
            decoder_filters = decoder_filters,
            encoder_method = encoder_method,
            decoder_method = decoder_method,
            center_filter = center_filter, 
            decode_try_upsample_first = decode_try_upsample_first, 
            encode_try_maxpool_first = encode_try_maxpool_first,
            merge_method = merge_method)

    
    try:
        task_max = int(os.environ['SLURM_ARRAY_TASK_MAX'])
        print(f"Task: {index}/{len(iterate)-1} \t(Task_Max: {task_max})")
    except:
        print(f"Task: {index}/{len(iterate)-1}")







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

def test_run(asd):

    for index in range(2):
        backbone_name = "vgg16"
        encoder_weights = None
        encoder_freeze = False
        activation = "sigmoid"
        decoder_block_type = "upsampling"
        encoder_freeze = False

        modality_name = ["ADC","t2tsetra"]       
        autoencoder_learning_rate = [1e-3, 1e-4]
        cube = [[[15,15],[60,60],100],[[10,20],[40,60],50]]

        
        center_filter = 256
        decoder_filters = (256, 128, 64, 32, 16)
        encoder_method, decoder_method, encode_try_maxpool_first, decode_try_upsample_first = ["upsample","maxpool",False,True]
        


        iterate = list(itertools.product(
                                cube,
                                autoencoder_learning_rate,
                                modality_name
                                ))

        cube_params, autoencoder_learning_rate, modality_name = iterate[index]
        minmax_shape_reduction, minmax_augmentation_percentage,mask_vs_rotation_percentage = cube_params

        if modality_name == "ADC":
            autoencoder_epocs = 500
            autoencoder_batchsize = 32
            reshape_dim = (32,128,96)
        elif modality_name == "t2tsetra":
            autoencoder_epocs = 250
            autoencoder_batchsize = 2
            reshape_dim = None

        # job_name = f"e{autoencoder_epocs}_lr{autoencoder_learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_bs{autoencoder_batchsize}_imagenet{encoder_weights}"

        job_name=f"e{autoencoder_epocs}_lr{autoencoder_learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_bs{autoencoder_batchsize}_em{encoder_method}_dm{decoder_method}_cf{center_filter}_df{decoder_filters[0]}-{decoder_filters[-1]}_etmf{encode_try_maxpool_first}_dtuf{decode_try_upsample_first}_imagenet{encoder_weights}"

        parameters.set_global(
                data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
                job_name = job_name,
                autoencoder_batchsize = autoencoder_batchsize,
                encoder_freeze = encoder_freeze,
                decoder_block_type = decoder_block_type, 
                autoencoder_epocs = autoencoder_epocs,
                autoencoder_learning_rate  = autoencoder_learning_rate,
                minmax_shape_reduction  = minmax_shape_reduction,
                minmax_augmentation_percentage  = minmax_augmentation_percentage,
                mask_vs_rotation_percentage = mask_vs_rotation_percentage,
                encoder_weights = encoder_weights
                )

        parameters.add_modality(
            modality_name = modality_name, 
            reshape_dim=reshape_dim,  
            autoencoder_batchsize=autoencoder_batchsize,
            skip_modality = True
            )


        
        autoencoder_batchsize = 1
        autoencoder_epocs = 350

        # job_name = f"e{autoencoder_epocs}_lr{autoencoder_learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_bs{autoencoder_batchsize}_em{encoder_method}_dm{decoder_method}_cf{center_filter}_df{decoder_filters[0]}-{decoder_filters[-1]}_etmf{encode_try_maxpool_first}_dtuf{decode_try_upsample_first}"

        job_name=f"e{autoencoder_epocs}_lr{autoencoder_learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_bs{autoencoder_batchsize}_em{encoder_method}_dm{decoder_method}_cf{center_filter}_df{decoder_filters[0]}-{decoder_filters[-1]}_etmf{encode_try_maxpool_first}_dtuf{decode_try_upsample_first}_imagenet{encoder_weights}"



        parameters.set_global(
                data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
                job_name = job_name,
                autoencoder_batchsize = autoencoder_batchsize,
                encoder_freeze = encoder_freeze,
                decoder_block_type = decoder_block_type, 
                autoencoder_epocs = autoencoder_epocs,
                autoencoder_learning_rate  = autoencoder_learning_rate,
                minmax_shape_reduction  = minmax_shape_reduction,
                minmax_augmentation_percentage  = minmax_augmentation_percentage,
                mask_vs_rotation_percentage = mask_vs_rotation_percentage,
                encoder_weights = encoder_weights
                )

        parameters.join_modalities(["ADC", "t2tsetra"], encoder_method = encoder_method, decoder_method=decoder_method, center_filter=center_filter, decoder_filters=decoder_filters, decode_try_upsample_first=decode_try_upsample_first,encode_try_maxpool_first=encode_try_maxpool_first)

def single_run_ssFALSE(index):
    
    self_superviced = False
    #encoder
    encoder_weights = [None,"imagenet"]

    #test
    classifier_freeze_encoder = [False]#[False,True] #IF TIME
    classifier_multi_dense = [False]#[False,True]    #IF TIME
    
    #eval
    bootpercentage = [1,0.1,0.2,0.5]
    #16

    iterate = list(itertools.product(
                            bootpercentage,
                            classifier_freeze_encoder,
                            classifier_multi_dense,
                            encoder_weights
                            ))

    bootpercentage,classifier_freeze_encoder,classifier_multi_dense,encoder_weights = iterate[index]

    parameters.set_global(
            data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
            encoder_weights = encoder_weights,
            self_superviced = self_superviced,

            bootpercentage = bootpercentage,
            classifier_freeze_encoder = classifier_freeze_encoder,
            classifier_multi_dense = classifier_multi_dense
            )


    parameters.add_modality(
        modality_name = "ADC", 
        reshape_dim=(32,128,96),  
        autoencoder_batchsize = 32,
        autoencoder_epocs = 80,
        skip_modality=False, 
        classifier_train_batchsize = 2,
        classifier_train_epochs = 35,
        classifier_test_batchsize = 2
        )
    parameters.add_modality(
        modality_name = "t2tsetra", 
        reshape_dim=None,  
        autoencoder_batchsize = 2,
        autoencoder_epocs = 80,
        skip_modality=False, 
        classifier_train_batchsize = 2,
        classifier_train_epochs = 35,
        classifier_test_batchsize = 2
        )
        


    try:
        task_max = int(os.environ['SLURM_ARRAY_TASK_MAX'])
        print(f"Task: {index}/{len(iterate)-1} \t(Task_Max: {task_max})")
    except:
        print(f"Task: {index}/{len(iterate)-1}")



def merged_run_ssFALSE(index):


    self_superviced = False
    #autoencoder
    encoder_weights = [None,"imagenet"]
    
    center_filter = 512
    decoder_filters = (256, 128, 64, 32, 16)
    updown = [["upsample","maxpool",False,True],["maxpool","upsample",True,True]]
    merge_method = ["concat","avg","add","max","multiply"]

    #test
    classifier_freeze_encoder = [False]#[False,True]
    classifier_multi_dense = [False]#[False,True]
    
    #eval
    bootpercentage = [1,0.1,0.2,0.5]


    iterate = list(itertools.product(
                        bootpercentage,
                        classifier_freeze_encoder,
                        classifier_multi_dense,
                        encoder_weights,
                        merge_method,
                        updown
                        ))

    bootpercentage,classifier_freeze_encoder,classifier_multi_dense,encoder_weights,merge_method, updown = iterate[index]

    encoder_method, decoder_method, encode_try_maxpool_first, decode_try_upsample_first = updown


    parameters.set_global(
            data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
            encoder_weights = encoder_weights,
            self_superviced = self_superviced,

            autoencoder_epocs = 80,
            autoencoder_batchsize = 1,
            classifier_train_epochs = 35,
            classifier_train_batchsize = 2, #TEST MED 2 +++
            classifier_test_batchsize = 2,  #TEST MED 2 +++

            bootpercentage = bootpercentage,
            classifier_freeze_encoder = classifier_freeze_encoder,
            classifier_multi_dense = classifier_multi_dense,

            
            )


    parameters.add_modality(
        modality_name = "ADC", 
        reshape_dim=(32,128,96),
        skip_modality=True
        )
    parameters.add_modality(
        modality_name = "t2tsetra", 
        reshape_dim=None,
        skip_modality=True
        )
        

    parameters.join_modalities(["ADC", "t2tsetra"],
            decoder_filters = decoder_filters,
            encoder_method = encoder_method,
            decoder_method = decoder_method,
            center_filter = center_filter, 
            decode_try_upsample_first = decode_try_upsample_first, 
            encode_try_maxpool_first = encode_try_maxpool_first,
            merge_method = merge_method)

    
    try:
        task_max = int(os.environ['SLURM_ARRAY_TASK_MAX'])
        print(f"Task: {index}/{len(iterate)-1} \t(Task_Max: {task_max})")
    except:
        print(f"Task: {index}/{len(iterate)-1}")

