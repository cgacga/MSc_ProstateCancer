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
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
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



# Seed for reproducibility
set_seed()


def main(*args, **kwargs):
    start_time = time.time()
    
    parameters.set_global(
            data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
            encoder_weights = None,
            self_superviced = True,

            autoencoder_epocs = 300,
            autoencoder_batchsize = 2,
            classifier_train_epochs = 30,
            classifier_train_batchsize = 1,
            classifier_test_batchsize = 1, 
            classifier_train_learning_rate = 1e-5,
            classifier_test_learning_rate = 1e-5,

            bootpercentage = 1,
            classifier_freeze_encoder = True,
            classifier_multi_dense = False,

            minmax_shape_reduction  = [15,15],
            minmax_augmentation_percentage  = [60,60],
            mask_vs_rotation_percentage = 50
            )


    parameters.add_modality(
        modality_name = "ADC", 
        reshape_dim=(32,128,96),
        skip_modality=False,
        autoencoder_epocs = 500
        )
    parameters.add_modality(
        modality_name = "t2tsetra", 
        reshape_dim=None,
        skip_modality=False
        )
        

    parameters.join_modalities(["ADC", "t2tsetra"],
            decoder_filters = (256, 128, 64, 32, 16),
            encoder_method = "maxpool",
            decoder_method = "upsample",
            center_filter = 512, 
            decode_try_upsample_first = True, 
            encode_try_maxpool_first = True,
            merge_method = "avg")

            


    pat_slices, pat_df = preprocess(parameters)


    for modality_name in parameters.lst.keys():
        parameters.set_current(modality_name)
        if modality.skip_modality:
            continue
        print(modality.job_name)
            
        encoder = None

        if os.path.isdir(modality.AE_path) :
            try:
                encoder = tf.keras.models.load_model(modality.AE_path, compile=False)
                print(f"Loaded Encoder from {modality.AE_path}")
            except:
                pass

        if encoder:
            y_train, y_val, y_test, patients_df = split_data(pat_slices, pat_df, autoencoder = False)

            labels = {}
            for split in patients_df.split.unique():

                
                label_split = patients_df.sort_values("pat_idx").drop_duplicates(["Subject ID", "Study Date"]).ClinSig.where(patients_df.split == split).dropna().replace({"non-significant": 0, "significant": 1})

                label_split = pd.get_dummies(pd.Series(label_split))

                labels[split] = tf.constant(label_split, dtype=tf.int32)


            classifier = build_train_classifier(encoder, y_train[modality.idx], y_val[modality.idx], labels)

            evaluate_classifier(classifier, y_test[modality.idx], labels["y_test"])
            
            print(f"\nCurrent parameters:\n{modality.mrkdown()}")
            

        else:
            
            y_train, y_val, patients_df = split_data(pat_slices, pat_df, autoencoder = True)
            print(f"\nCurrent parameters:\n{modality.mrkdown()}")
        
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
        args = [x for x in sys.argv if '=' not in x]
        kwargs = {x.split('=')[0]: x.split('=')[1] for x in sys.argv if '=' in x}        
        
        main(*args, **kwargs)
    else: main()

