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


    classifier_freeze_encoder = [False,True]
    classifier_multi_dense = [False,True]
    #classifier_train_batchsize = [2,8,32]#16,
    classifier_train_epochs = [25]#[100,200]#100,
    classifier_train_learning_rate = [1e-5]#[1e-4,1e-5]
    


    iterate = list(itertools.product(
                            cube,
                            learning_rate,
                            modality_name,
                            classifier_freeze_encoder,
                            classifier_multi_dense,
                            classifier_train_epochs,
                            classifier_train_learning_rate
                            ))

    cube_params, learning_rate, modality_name,classifier_freeze_encoder,classifier_multi_dense,classifier_train_epochs,classifier_train_learning_rate = iterate[index]
    minmax_shape_reduction, minmax_augmentation_percentage,mask_vs_rotation_percentage = cube_params

    if modality_name == "ADC":
        epochs = 500
        batch_size = 32
        classifier_train_batchsize = 32
        reshape_dim = (32,128,96)
    elif modality_name == "t2tsetra":
        epochs = 250
        batch_size = 2
        classifier_train_batchsize = 2
        reshape_dim = None

    autoencoder_job_name = f"e{epochs}_lr{learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_bs{batch_size}"


    classifier_freeze_encoder = classifier_freeze_encoder
    classifier_multi_dense = classifier_multi_dense
    classifier_train_batchsize = classifier_train_batchsize
    classifier_train_epochs = classifier_train_epochs
    classifier_test_learning_rate = classifier_train_learning_rate
    classifier_test_batchsize = classifier_train_batchsize

    job_name = f"e{epochs}_lr{learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_bs{batch_size}_cfe{classifier_freeze_encoder}_cdm{classifier_multi_dense}_ctrab{classifier_train_batchsize}_cte{classifier_train_epochs}_ctralr{classifier_train_learning_rate}_ctesb{classifier_test_batchsize}_cteslr{classifier_test_learning_rate}"

    parameters.set_global(
            data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
            job_name = job_name,
            encoder_freeze = encoder_freeze,
            decoder_block_type = decoder_block_type, 
            minmax_shape_reduction  = minmax_shape_reduction,
            minmax_augmentation_percentage  = minmax_augmentation_percentage,
            mask_vs_rotation_percentage = mask_vs_rotation_percentage,
            autoencoder_job_name = autoencoder_job_name,
            classifier_freeze_encoder = classifier_freeze_encoder,
            classifier_multi_dense = classifier_multi_dense,
            classifier_train_batchsize = classifier_train_batchsize,
            classifier_train_epochs = classifier_train_epochs,
            classifier_test_learning_rate = classifier_test_learning_rate,
            classifier_test_batchsize = classifier_test_batchsize,
            autoencoder_batchsize = batch_size,
            autoencoder_learning_rate = learning_rate,
            autoencoder_epocs = epochs
            )

    parameters.add_modality(
        modality_name = modality_name, 
        reshape_dim=reshape_dim#,  
        #batch_size=batch_size
        )

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

    learning_rate = [1e-4]#[1e-3, 1e-4]
    cube = [[[15,15],[60,60],100],[[10,20],[40,60],50]]

    updowm = [["maxpool","upsample",False,True],["upsample","maxpool",False,True]]#,["maxpool","upsample",True,True]]

    classifier_freeze_encoder = [False,True]
    classifier_multi_dense = [False,True]
    #classifier_train_batchsize = [1,2,8],#16,
    #classifier_train_epochs = [100,200]#100,
    classifier_train_learning_rate = [1e-4,1e-5]


    classifier_train_batchsize = 1
    classifier_train_epochs = 20
    

    # rm /e350_lr0.001_sr15-15_ap60-60_mvsrp100_efFalse_bs1_emupsample_dmmaxpool_cf1024_df1024-64_etmfFalse_dtufTrue/Merged_ADC-t2tsetra/
    # rm /e350_lr0.001_sr15-15_ap60-60_mvsrp100_efFalse_bs1_emmaxpool_dmupsample_cf1024_df1024-64_etmfFalse_dtufTrue/Merged_ADC-t2tsetra/
    # test 128 istedenfor 1024
    filters = [[512,(256, 128, 64, 32, 16)],[1024,(512, 256, 128, 64, 32)],[256,(256, 128, 64, 32, 16)]]#,[512,(512,256, 128, 64, 32)]

    iterate = list(itertools.product(
                            
                            filters,                        
                            learning_rate,
                            updowm,
                            cube,
                            classifier_freeze_encoder,
                            classifier_multi_dense,
                            #classifier_train_batchsize,
                            #classifier_train_epochs,
                            classifier_train_learning_rate
                            ))



    # for i in range(len(iterate)):
    #     print(iterate[i])

    filters,learning_rate, updowm,cube_params, classifier_freeze_encoder, classifier_multi_dense,  classifier_train_learning_rate = iterate[index]
    center_filter,decoder_filters = filters
    minmax_shape_reduction, minmax_augmentation_percentage,mask_vs_rotation_percentage = cube_params
    encoder_method, decoder_method, encode_try_maxpool_first, decode_try_upsample_first = updowm

        
    autoencoder_job_name = f"e{epochs}_lr{learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_bs{batch_size}_em{encoder_method}_dm{decoder_method}_cf{center_filter}_df{decoder_filters[0]}-{decoder_filters[-1]}_etmf{encode_try_maxpool_first}_dtuf{decode_try_upsample_first}"


    classifier_freeze_encoder = classifier_freeze_encoder
    classifier_multi_dense = classifier_multi_dense
    classifier_train_batchsize = classifier_train_batchsize
    classifier_train_epochs = classifier_train_epochs
    classifier_test_learning_rate = classifier_train_learning_rate
    classifier_test_batchsize = classifier_train_batchsize

    job_name = f"e{epochs}_lr{learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_bs{batch_size}_em{encoder_method}_dm{decoder_method}_cf{center_filter}_df{decoder_filters[0]}-{decoder_filters[-1]}_etmf{encode_try_maxpool_first}_dtuf{decode_try_upsample_first}_cfe{classifier_freeze_encoder}_cdm{classifier_multi_dense}_ctrab{classifier_train_batchsize}_cte{classifier_train_epochs}_ctralr{classifier_train_learning_rate}_ctesb{classifier_test_batchsize}_cteslr{classifier_test_learning_rate}"




    parameters.set_global(
                data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
                job_name = job_name,
                encoder_freeze = encoder_freeze,
                decoder_block_type = decoder_block_type, 
                autoencoder_epocs = epochs,
                minmax_shape_reduction  = minmax_shape_reduction,
                minmax_augmentation_percentage  = minmax_augmentation_percentage,
                mask_vs_rotation_percentage = mask_vs_rotation_percentage,
                autoencoder_job_name = autoencoder_job_name,
            classifier_freeze_encoder = classifier_freeze_encoder,
            classifier_multi_dense = classifier_multi_dense,
            classifier_train_batchsize = classifier_train_batchsize,
            classifier_train_epochs = classifier_train_epochs,
            classifier_test_learning_rate = classifier_test_learning_rate,
            classifier_test_batchsize = classifier_test_batchsize,
            autoencoder_batchsize = batch_size,
            autoencoder_learning_rate = learning_rate
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
    
    # model = get_merged_model()

    # if not os.path.exists(modality.model_path):
    #     os.makedirs(modality.model_path)

    # tf.keras.utils.plot_model(
    # model,
    # show_shapes=True,
    # show_layer_activations = True,
    # expand_nested=True,
    # to_file=os.path.abspath(modality.model_path+f"autoencoder.png")
    # )

    # tf.keras.backend.clear_session()




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
