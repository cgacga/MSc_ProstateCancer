### Importing libraries ###
import os, sys, time, random, gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from preprocess import *
from data_augmentation import *
from model_building import *
from img_display import *
from slurm_array import *
from params import *


### GPU Cluster setup ###

# IMPORTANT NOTE FOR USERS RUNNING UiS GPU CLUSTERS:
# This script is made to be used with slurm workload manager. The slurm scheduling system will automatically assign gpus and this script will use all awailable GPU's. This script should not be run outside of slurm wihout changing parameters for building the model and compiling it.



# By setting config.gpu_options.allow_growth to True, Tensorflow will only grab as much GPU-memory as needed. If additional memory is required later in the code, Tensorflow will allocate more memory as needed. This allows the user to run two or three programs on the same GPU.
# tf.keras.backend.clear_session()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    # logical_gpus = tf.config.list_logical_devices('GPU')
    # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# session = tf.compat.v1.Session(config=config)

# tf.compat.v1.keras.backend.set_session(session)

# session = tf.compat.v1.InteractiveSession(config=config)

# K.set_session(session)


# The variable TF_CPP_MIN_LOG_LEVEL sets the debugging level of Tensorflow and controls what messages are displayed onscreen. Defaults value is set to 0, so all logs are shown. Set TF_CPP_MIN_LOG_LEVEL to 1 to filter out INFO logs, 2 to additional filter out WARNING, and 3 to additionally filter out ERROR. Disable debugging information from tensorflow. 
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
# TF_CPP_MIN_VLOG_LEVEL brings in extra debugging information, and in reverse. Its default value is 0 and as it increases, more debugging messages are logged in.
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'


# Deprecation removes deprecated warning messages
# from tensorflow.python.framework import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = True

# Seed for reproducibility
# np.random.seed(42)

# For reproducible results    
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(s) 
set_seed(42)


def main(*args, **kwargs):
    start_time = time.time()
    
    if "slurm_array" in kwargs:
        slurm_array = kwargs["slurm_array"]
    #if "index" in kwargs:
        index = int(kwargs["index"])

        if slurm_array == "single":
            single_run(index)
        elif slurm_array == "merged":
            merged_run(index)
        elif slurm_array == "samesize":
            samesize_run(index)
    else:
        raise ValueError("Missing slurm_array in kwargs")





    # try:
    #     #print(f"SLURM_ARRAY_TASK_ID - {os.environ['SLURM_ARRAY_TASK_ID']}")
        
    #     task_parameters()
    # except Exception as e:
    #     #print(e)
    #     #sys.exit("Excepted SLURM_ARRAY_TASK_ID to be set.")
    #     #task_idx = os.environ['SLURM_JOB_ID']
            
    #     epochs = 500
    #     batch_size = 1
    #     encoder_weights = "imagenet"
    #     encoder_freeze = False
    #     no_adjacent = False
               
    #     backbone_name = "vgg16"
    #     activation = "sigmoid"
    #     decoder_block_type = "upsampling"#["upsampling", "transpose"]
    #     #learning_rate = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    #     learning_rate = 1e-4

    #     minmax_augmentation_percentage  = [40,60]#[60,60]
    #     minmax_shape_reduction  = [10,20]#[15,15]
    #     mask_vs_rotation_percentage = 50#100
    

    #     #up = ["upsample","transpose","padd"]
    #     #down = ["maxpool","avgpool","reshape", "crop"]
    #     encoder_method = "upsample"#"maxpool"
    #     decoder_method = "maxpool"#"upsample"
    #     center_filter = 1024
    #     decoder_filters = (256, 128, 64, 32, 16)


    #     decode_try_upsample_first = True #False
    #     encode_try_maxpool_first  = True #False

    #     job_name = f"NOTnormalized_{backbone_name}_{activation}_{decoder_block_type}_e{epochs}_lr{learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_bs{batch_size}_em{encoder_method}_dm{decoder_method}_cf{center_filter}_df{decoder_filters[0]}-{decoder_filters[-1]}_etmf{encode_try_maxpool_first}_dtuf{decode_try_upsample_first}"
    #     #_{encoder_weights}_


    #     parameters.set_global(
    #         data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
    #         job_name = job_name,
    #         backbone_name = backbone_name,
    #         activation = activation,
    #         batch_size = batch_size,
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

    #     parameters.add_modality(
    #         modality_name = "ADC", 
    #         reshape_dim=(32,128,96),
    #         batch_size=32, 
    #         skip_modality=False
    #         )
    #     # parameters.add_modality(
    #     #     modality_name = "t2tsetra", 
    #     #     #reshape_dim=None,  
    #     #     reshape_dim=(32,384,384),
    #     #     batch_size=2,
    #     #     skip_modality=False
    #     #     )


    #     # parameters.join_modalities(["ADC", "t2tsetra"], encoder_method = encoder_method, decoder_method=decoder_method, center_filter=center_filter, decoder_filters=decoder_filters, decode_try_upsample_first=decode_try_upsample_first,encode_try_maxpool_first=encode_try_maxpool_first)

    #     # parameters.set_current("Merged")
    #     # get_merged_model()
    #     # tf.keras.backend.clear_session()
            
    y_train, y_val, pat_df = preprocess(parameters)
    
    # if os.path.isdir(modality.model_path):
    #     print("Loading model")
    #     encoder = tf.keras.models.load_model(modality.model_path)
    # else:


    #models = {}
    # for modality_name in parameters.lst:
    #     parameters.set_current(modality_name)

    for modality_name in parameters.lst.keys():
        parameters.set_current(modality_name)
        if modality.skip_modality:
            continue
        print(f"\nCurrent parameters:\n{modality.mrkdown()}")

        trainDS, valDS = augment_build_datasets(y_train[modality.idx], y_val[modality.idx])

        autoencoder = model_building(trainDS, valDS)

        encoder = Model(autoencoder.input, autoencoder.get_layer("center_block2_relu").output,name=f'encoder_{modality.modality_name}')

        encoder.save(modality.model_path)

        tf.keras.utils.plot_model(
        encoder,
        show_shapes=True,
        show_layer_activations = True,
        expand_nested=True,
        to_file=os.path.abspath(modality.model_path+f"encoder.png")
        )
        
        #models[modality_name] = autoencoder
        del autoencoder, encoder, trainDS, valDS
        tf.keras.backend.clear_session()
        gc.collect()
    
    print("\n"+f"Job completed {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '*')+"\n")
 



    
    #data_path = "../data/manifest-A3Y4AE4o5818678569166032044/"
    
    #tags = {"ADC": (32,128,96),"t2tsetra": None} 
    #tags = {"t2tsetra": None} 
    #tags = {"ADC": (32,128,96)} 

    #y_train, y_val, pat_df = preprocess(data_path,tags)
    # x_train, x_test, x_val, y_train, y_val, y_val = data_augmentation(pat_slices, pat_df)

    # x_train, _, _, x_train_noisy, _, _ = data_augmentation(pat_slices, pat_df)
    # del pat_slices



    #batchsize = {"ADC": 32, "t2tsetra": 2}
    #batchsize = {"ADC": 32, "t2tsetra": 2}






    #models = {}
    #for modality in tags.keys():
        # modelpath = f"../models/{modality}/{os.environ['SLURM_JOB_NAME']}/{os.environ['SLURM_JOB_ID']}-{os.environ['SLURM_JOB_NAME']}"
        #modelpath = f"../models/{modality}/{os.environ['SLURM_JOB_NAME']}"
        #os.path.join(modelpath,f"multiple_05_{modality}"))

#        shape,idx = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality, case=False)].values[0]
        # train_data = tf.data.Dataset.from_tensor_slices((x_train_noisy[idx], x_train[idx]))
        # val_data = tf.data.Dataset.from_tensor_slices((x_test_noisy[idx], x_test[idx]))
        # models[modality] = model_building(shape, modelpath, train_data, val_data)
        #https://stackoverflow.com/questions/52724022/model-fits-on-a-single-gpu-but-script-crashes-when-trying-to-fit-on-multiple-gpu

        #model = model_building(shape, modelpath, x_train[idx],y_train[idx], x_test[idx], y_val[idx])

        

        # trainDS, valDS = augment_build_datasets(y_train[idx], y_val[idx], batchsize[modality])

        # model = model_building(shape, modality, trainDS, valDS)
 

        #sample = y_val[idx][0:5] #0:1 

        #aug_sample = augment_patches(sample)
                        
        #gpus = tf.config.list_physical_devices('GPU')
        #strategy = tf.distribute.MirroredStrategy()
        #if len(gpus)>1:
        #    with strategy.scope():
        #        pred_sample = model.predict(tf.repeat(augment_patches(aug_sample),3,-1))
        #else: pred_sample = model.predict(tf.repeat(augment_patches(aug_sample),3,-1))
        
        #img_stack = np.stack([aug_sample,pred_sample],0+1).reshape(*(-(0==j) or s for j,s in enumerate(pred_sample.shape)))

        #savepath = f"../models/{os.environ['SLURM_JOB_NAME']}/{modality}/{os.environ['SLURM_JOB_NAME']}_{modality}"
        # savepath = f"../models/{modality}/{os.environ['SLURM_JOB_NAME']}/{os.environ['SLURM_JOB_NAME']}_{modality}"

        #savepath = f"../models/{os.environ['SLURM_JOB_NAME']}/{modality}/{os.environ['SLURM_JOB_ID']}/{os.environ['SLURM_JOB_NAME']}_{modality}"

        #savepath = f"{os.environ['SLURM_JOB_NAME']}/{modality}/{os.environ['SLURM_JOB_ID']}/{os.environ['SLURM_JOB_NAME']}_{modality}"

        


        #img_pltsave([img for img in img_stack],savepath)
        #os.path.join(modelpath,f"multiple_05_{modality}"))

        

        # trainDS, valDS = augment_build_datasets(y_train[idx], y_val[idx], batchsize[modality])

        # model = model_building(shape, modality, trainDS, valDS)

        # models[modality] = model
        # del model
        # del trainDS
        # del valDS
        # tf.keras.backend.clear_session()
        # gc.collect()

        # loss, acc = models[modality].evaluate(x_test_noisy[idx], x_test[idx], verbose=2)
        
        # print(f"Test Loss {loss}\nTest Acc {acc}")
        
        # loss, acc = models[modality].evaluate(x_val_noisy[idx], x_val[idx], verbose=2)

        # print(f"Validation Loss {loss}\n Validation Acc {acc}")
        
    

    #     trainDS, valDS = augment_build_datasets(y_train[idx], y_val[idx], batchsize[modality])

    #     model = model_building(shape, modality, trainDS, valDS)
        
    #     models[modality] = model
    #     del model
    #     del trainDS
    #     del valDS
    #     tf.keras.backend.clear_session()
    #     gc.collect()
    
    # print("\n"+f"Job completed {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '*')+"\n")

    #test_result = t2_model.evaluate(x_test,y_val, verbose = 1)
    #print("Test accuracy :\t", round (test_result[1], 4))

   

    # predictions = t2_model.predict(x_test_noisy[1])
    # display([x_test_noisy[1], predictions])

    # predictions = t2_model.predict(x_val[1])
    # display([x_val[1], predictions])


    # predictions = t2_model.predict(x_val_noisy[1])
    # display([x_val_noisy[1], predictions])

    # predictions = t2_model.predict(x_val_noisy[1])
    # display([x_val_noisy[1], predictions])
    
    # keras.backend.clear_session()
    #return #pat_df, x_train, x_test, x_val, x_train_noisy, x_test_noisy



if __name__ == '__main__':
    try:
        print(f"SLURM_JOB_NAME - {os.environ['SLURM_JOB_NAME']}")
        print(f"SLURM_JOB_ID - {os.environ['SLURM_JOB_ID']}")
    except:
        pass
    print(f"Tensorflow version - {tf.__version__}")
    print(f"GPUs Available: {tf.config.list_physical_devices('GPU')}") 
    print()
    
    
    if len(sys.argv)>1:
        #print(sys.argv)
        #kwargs={kw[0]:kw[1] for kw in [ar.split('=') for ar in sys.argv if ar.find('=')>0]}
        #import json
        #kwargs=json.loads(sys.argv[1])
        args = [x for x in sys.argv if '=' not in x]
        kwargs = {x.split('=')[0]: x.split('=')[1] for x in sys.argv if '=' in x}        
        #print Func(*args, **kwargs)  

        #print(f"kwargs = {kwargs}")

        # main(**kwargs)
        main(*args, **kwargs)
    else: main()

