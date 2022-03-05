#%%
### Importing libraries ###
import os
import time
import sys
import tensorflow as tf
import numpy as np


from data_augmentation import *
from preprocess import *
from model_building import *
from img_display import *

### GPU Cluster setup ###

# IMPORTANT NOTE FOR USERS RUNNING UiS GPU CLUSTERS:
# This script is made to be used with slurm workload manager. The slurm scheduling system will automatically assign gpus and this script will use all awailable GPU's. This script should not be run outside of slurm wihout changing parameters for building the model and compiling it.


# The variable TF_CPP_MIN_LOG_LEVEL sets the debugging level of Tensorflow and controls what messages are displayed onscreen. Defaults value is set to 0, so all logs are shown. Set TF_CPP_MIN_LOG_LEVEL to 1 to filter out INFO logs, 2 to additional filter out WARNING, and 3 to additionally filter out ERROR. Disable debugging information from tensorflow. 
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
# TF_CPP_MIN_VLOG_LEVEL brings in extra debugging information, and in reverse. Its default value is 0 and as it increases, more debugging messages are logged in.
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'

# By setting config.gpu_options.allow_growth to True, Tensorflow will only grab as much GPU-memory as needed. If additional memory is required later in the code, Tensorflow will allocate more memory as needed. This allows the user to run two or three programs on the same GPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)




# Deprecation removes deprecated warning messages
# from tensorflow.python.framework import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = True

# Seed for reproducibility
np.random.seed(42)


def main(**kwargs):
    start_time = time.time()
    data_path = "../data/manifest-A3Y4AE4o5818678569166032044/"
    #tags = {"ADC": None,"t2tsetra": (320,320,20)} 
    tags = {"t2tsetra": (320,320,20)} 

    pat_slices, pat_df = preprocess(data_path,tags,False)
    x_train, x_test, x_val, x_train_noisy, x_test_noisy, x_val_noisy = data_augmentation(pat_slices, pat_df)
    # x_train, _, _, x_train_noisy, _, _ = data_augmentation(pat_slices, pat_df)
    del pat_slices

    models = {}
    for modality in tags.keys():
        modelpath = f"../models/{modality}/{os.environ['SLURM_JOB_NAME']}/{os.environ['SLURM_JOB_ID']}-{os.environ['SLURM_JOB_NAME']}"
        shape,idx = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality, case=False)].values[0]
        train_data = tf.data.Dataset.from_tensor_slices((x_train_noisy[idx], x_train[idx]))
        val_data = tf.data.Dataset.from_tensor_slices((x_test_noisy[idx], x_test[idx]))
        models[modality] = model_building(shape, modelpath, train_data, val_data)

        #test_save
        img_pltsave([x_train_noisy[idx][0],x_train[idx][0]],modelpath+"test_saving.png")
        
        predictions = models[modality].predict(x_test_noisy[idx][0])
        img_pltsave([x_test[idx][0], x_test_noisy[idx][0], predictions],modelpath+"test.png")
        loss, acc = models[modality].evaluate(x_test_noisy[0], x_train[0], verbose=2)
        
        print(f"Test Loss {loss}\nTest Acc {acc}")
        
        predictions = models[modality].predict(x_val_noisy[idx][0])
        img_pltsave([x_val[idx][0], x_val_noisy[idx][0], predictions],modelpath+"validation.png")

        loss, acc = models[modality].evaluate(x_val_noisy[0], x_val[0], verbose=2)

        print(f"Validation Loss {loss}\n Validation Acc {acc}")
        
    
    

    #test_result = t2_model.evaluate(x_test,y_test, verbose = 1)
    #print("Test accuracy :\t", round (test_result[1], 4))

   

    # predictions = t2_model.predict(x_test_noisy[1])
    # display([x_test_noisy[1], predictions])

    # predictions = t2_model.predict(x_val[1])
    # display([x_val[1], predictions])


    # predictions = t2_model.predict(x_val_noisy[1])
    # display([x_val_noisy[1], predictions])

    # predictions = t2_model.predict(x_val_noisy[1])
    # display([x_val_noisy[1], predictions])
    print("\n"+f"Completed {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '*')+"\n")
    # keras.backend.clear_session()
    return #pat_df, x_train, x_test, x_val, x_train_noisy, x_test_noisy


# df, x_train, x_test, x_val, x_train_noisy, x_test_noisy = main()

if __name__ == '__main__':
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")    
    print(f"SLURM_JOB_NAME - {os.environ['SLURM_JOB_NAME']}")
    print(f"SLURM_JOB_ID - {os.environ['SLURM_JOB_ID']}")
    
    if len(sys.argv)>1:
        print(sys.argv)
        kwargs={kw[0]:kw[1] for kw in [ar.split('=') for ar in sys.argv if ar.find('=')>0]}
        print(f"kwargs = {kwargs}")
        # main(**kwargs)
        main()
    else: main()

