#%%
### Importing libraries ###
import os, sys, time, random, gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from preprocess import *
from data_augmentation import *
#import data_augmentation
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

set_seed()


def main(*args, **kwargs):
    start_time = time.time()
    
    if "slurm_array" in kwargs:
        slurm_array = kwargs["slurm_array"]
    #if "index" in kwargs:
        index = int(kwargs["index"])

        if slurm_array == "all":
            all(index)
        elif slurm_array == "single":
            single_run(index)
        elif slurm_array == "merged":
            # if index >96:
            #   i = index - 97
            #   single_run(i)
            # elif index == 96:
            #   test_run(index)
            # else:
            #   merged_run(index)
            
            merged_run(index)
            # if index == 96:
            #   test_run(index)
            # else:
            #   merged_run(index)
        elif slurm_array == "samesize":
            samesize_run(index)

        elif slurm_array == "singlefalse":
            single_run_ssFALSE(index)
        elif slurm_array == "mergedfalse":
            merged_run_ssFALSE(index)
    else:
        #raise ValueError("Missing slurm_array in kwargs")

            
        for index in range(2):
            
            encoder_weights = None #[None,"imagenet"]
            self_superviced = True #[True,False]
            
            classifier_train_epochs : int = 50#500

            #IKKE classifier_freeze_encoder
            #hvis self_superviced = False

            modality_name = ["ADC","t2tsetra"]       
            autoencoder_learning_rate = [1e-3, 1e-4]
            cube = [[[15,15],[60,60],100],[[10,20],[40,60],50]]

            center_filter = 1024
            decoder_filters = (256, 128, 64, 32, 16)
            encoder_method, decoder_method, encode_try_maxpool_first, decode_try_upsample_first = ["upsample","maxpool",False,True]
            #test *,*,True,*whatever]
            


            iterate = list(itertools.product(
                                    cube,
                                    autoencoder_learning_rate,
                                    modality_name
                                    ))

            cube_params, autoencoder_learning_rate, modality_name = iterate[index]
            minmax_shape_reduction, minmax_augmentation_percentage,mask_vs_rotation_percentage = cube_params


            

            if modality_name == "ADC":
                autoencoder_epocs = 35#100
                autoencoder_batchsize = 32
                reshape_dim = (32,128,96)
                skip_modality = False
                classifier_train_batchsize = 32
                classifier_train_epochs = 35
                classifier_test_batchsize = 32
                bootpercentage = 1
                merge_method = "avg"
            elif modality_name == "t2tsetra":
                autoencoder_epocs = 35#250
                autoencoder_batchsize = 2
                reshape_dim = None
                skip_modality = False
                classifier_train_batchsize = 2
                classifier_train_epochs = 35
                classifier_test_batchsize = 2
                bootpercentage = 1
                merge_method = "avg"

            job_name = f"ss{self_superviced}_ae{autoencoder_epocs}_abs{autoencoder_batchsize}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ctrab{classifier_train_batchsize}_cte{classifier_train_epochs}_ctesb{classifier_test_batchsize}_cteslr"



            #is not used when ss = False


            parameters.set_global(
                    data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
                    job_name = job_name,
                    autoencoder_learning_rate  = autoencoder_learning_rate,
                    minmax_shape_reduction  = minmax_shape_reduction,
                    minmax_augmentation_percentage  = minmax_augmentation_percentage,
                    mask_vs_rotation_percentage = mask_vs_rotation_percentage,
                    encoder_weights = encoder_weights,
                    self_superviced = self_superviced
                    )

            parameters.add_modality(
                job_name = job_name,
                modality_name = modality_name, 
                reshape_dim=reshape_dim,  
                autoencoder_batchsize = autoencoder_batchsize,
                autoencoder_epocs = autoencoder_epocs,
                skip_modality=skip_modality,
                classifier_train_batchsize = classifier_train_batchsize,
                classifier_train_epochs = classifier_train_epochs,
                classifier_test_batchsize = classifier_test_batchsize
                )



        classifier_train_batchsize : int = 2
        classifier_train_epochs : int = 35
        classifier_test_batchsize : int = 2
        autoencoder_batchsize = 1
        autoencoder_epocs = 35

        job_name = f"ss{self_superviced}_ae{autoencoder_epocs}_abs{autoencoder_batchsize}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvsrp{mask_vs_rotation_percentage}_ctrab{classifier_train_batchsize}_cte{classifier_train_epochs}_ctesb{classifier_test_batchsize}_cteslr"

        parameters.join_modalities(
                ["ADC", "t2tsetra"],
                encoder_method = encoder_method,
                decoder_method=decoder_method, 
                center_filter=center_filter, 
                decoder_filters=decoder_filters, 
                decode_try_upsample_first=decode_try_upsample_first,
                encode_try_maxpool_first=encode_try_maxpool_first,
                job_name = job_name,
                classifier_train_batchsize = classifier_train_batchsize,
                classifier_train_epochs = classifier_train_epochs,
                classifier_test_batchsize = classifier_test_batchsize,
                autoencoder_batchsize = autoencoder_batchsize,
                autoencoder_epocs = autoencoder_epocs)


    pat_slices, pat_df = preprocess(parameters)


    for modality_name in parameters.lst.keys():
        parameters.set_current(modality_name)
        if modality.skip_modality:
            continue

        print(modality.job_name)
            
        encoder = None
        classifier = None
        
        if os.path.isdir(modality.C_path):

            try:
                classifier = tf.keras.models.load_model(modality.C_path, compile=False)
                print(f"Loaded Classifier from {modality.C_path}")
            except:
                pass
        #if os.path.isdir(os.path.abspath(model_path+"/encoder/")) and not classifier:
        if os.path.isdir(modality.AE_path) and not classifier:
            try:
                encoder = tf.keras.models.load_model(modality.AE_path, compile=False)
                print(f"Loaded Encoder from {modality.AE_path}")
            except:
                pass

        
        if classifier:
            _, _, y_test, patients_df = split_data(pat_slices, pat_df, autoencoder = False)
            labels = {}
            for split in patients_df.split.unique():

                
                label_split = patients_df.sort_values("pat_idx").drop_duplicates(["Subject ID", "Study Date"]).ClinSig.where(patients_df.split == split).dropna().replace({"non-significant": 0, "significant": 1})

                labels[split] = tf.constant(label_split, dtype=tf.int32)
            evaluate_classifier(classifier, y_test[modality.idx], labels["y_test"])
        elif encoder:
            y_train, y_val, y_test, patients_df = split_data(pat_slices, pat_df, autoencoder = False)

            labels = {}
            for split in patients_df.split.unique():

                
                label_split = patients_df.sort_values("pat_idx").drop_duplicates(["Subject ID", "Study Date"]).ClinSig.where(patients_df.split == split).dropna().replace({"non-significant": 0, "significant": 1})

                labels[split] = tf.constant(label_split, dtype=tf.int32)


            classifier = build_train_classifier(encoder, y_train[modality.idx], y_val[modality.idx], labels)

            evaluate_classifier(classifier, y_test[modality.idx], labels["y_test"])
            
            print(f"\nCurrent parameters:\n{modality.mrkdown()}")
            

        else:
            
            y_train, y_val, patients_df = split_data(pat_slices, pat_df, autoencoder = True)
            #print(f"\nCurrent parameters:\n{modality.mrkdown()}")
        
            if modality.self_superviced:
                trainDS, valDS = augment_build_datasets(y_train[modality.idx], y_val[modality.idx])
            else:
                trainDS, valDS = None,None

            autoencoder = model_building(trainDS, valDS)

            encoder = Model(autoencoder.input, autoencoder.get_layer("center_block2_relu").output,name=f'encoder_{modality.modality_name}')


            encoder.save(modality.AE_path)

            tf.keras.utils.plot_model(
            encoder,
            show_shapes=True,
            show_layer_activations = True,
            expand_nested=True,
            to_file=os.path.abspath(modality.AE_path+f"encoder.png")
            )

            del y_train, y_val, trainDS, valDS, autoencoder
            tf.keras.backend.clear_session()
            gc.collect()

            y_train, y_val, y_test, patients_df = split_data(pat_slices, pat_df, autoencoder = False)

            labels = {}
            for split in patients_df.split.unique():

                label_split = patients_df.sort_values("pat_idx").drop_duplicates(["Subject ID", "Study Date"]).ClinSig.where(patients_df.split == split).dropna().replace({"non-significant": 0, "significant": 1})

                labels[split] = tf.constant(label_split, dtype=tf.int32)


            classifier = build_train_classifier(encoder, y_train[modality.idx], y_val[modality.idx], labels)

            evaluate_classifier(classifier, y_test[modality.idx], labels["y_test"])
            
            tf.keras.backend.clear_session()
            gc.collect()
        
    
    print("\n"+f"Job completed {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '*')+"\n")
 



if __name__ == '__main__':
    try:
        print(f"SLURM_JOB_NAME - {os.environ['SLURM_JOB_NAME']}")
        print(f"SLURM_JOB_ID - {os.environ['SLURM_JOB_ID']}")
    except:
        pass
    try:
        print(f"SLURM_ARRAY_JOB_ID - {os.environ['SLURM_ARRAY_JOB_ID']}")
        print(f"SLURM_ARRAY_TASK_ID - {os.environ['SLURM_ARRAY_TASK_ID']}")
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


# %%
