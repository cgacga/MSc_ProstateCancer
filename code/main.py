#%%
### Importing libraries ###
import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd

from data_augmentation import *
from preprocess import *
from model_building import *

### GPU Cluster setup ###

# IMPORTANT NOTE FOR USERS RUNNING UiS GPU CLUSTERS:
# IF USING SLURM, AVOID SPECIFY GPU (automatic discovery feature of slurm)

## The variable CUDA_VISIBLE_DEVICES will restrict what devices Tensorflow can see. By setting this to, e.g. 2, only GPU device #2 is available. The number used here should match the one used when reserved the GPU.
##os.environ['CUDA_VISIBLE_DEVICES'] = 'assigned gpu id'

# The variable TF_CPP_MIN_LOG_LEVEL sets the debugging level of Tensorflow and controls what messages are displayed onscreen. Defaults value is set to 0, so all logs are shown. Set TF_CPP_MIN_LOG_LEVEL to 1 to filter out INFO logs, 2 to additional filter out WARNING, and 3 to additionally filter out ERROR. Disable debugging information from tensorflow. 

#TF_CPP_MIN_VLOG_LEVEL brings in extra debugging information, and in reverse. Its default value is 0 and as it increases, more debugging messages are logged in.
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'

# By setting config.gpu_options.allow_growth to True, Tensorflow will only grab as much GPU-memory as needed. If additional memory is required later in the code, Tensorflow will allocate more memory as needed. This allows the user to run two or three programs on the same GPU.
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config = config)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)

# Deprecation removes deprecated warning messages
# from tensorflow.python.framework import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = True

# Seed for reproducibility
np.random.seed(42)

### Functions ###


### Main functions ###

def preprocess(data_path, tags):
    """
    Loads the slices from the data path, resamples the slices to the desired resolution, normalizes the
    slices and returns the resampled slices and the dataframe

    If size is set to None, then the most common size will be used
        Examples:
            tags = {"ADC":(86,128,20),"t2tsetra": (320,320,20)} 

            tags = {"T2TSETRA": (320,320,20), "adc": None} 

            tags = {"t2tsetra": None, "ADC": None}     
    
    :param data_path: path to the folder containing the data
    :param tags: a dictionary of tags and their corresponding sizes
    :return: the preprocessed slices and the dataframe with the metadata.
    """

    print(f"Preprocess started".center(50, '_'))
    start_time = time.time()

    pat_slices, pat_df = load_slices(data_path, tags)

    pat_slices, pat_df = resample_pat(pat_slices, pat_df, tags)

    pat_slices = normalize(pat_slices, pat_df)
        
    print("\n"+f"Preprocess finished {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

    return pat_slices, pat_df



def data_augmentation(pat_slices, pat_df):
    """
    This function takes in the patient slices and the patient dataframe and returns the train, test and
    validation data
    
    :param pat_slices: The list of slices that we have extracted from the patients
    :param pat_df: The dataframe containing the patient id's of the slices
    :return: the training, test and validation data sets.
    """

    print(f"Data augmentation started".center(50, '_'))
    start_time = time.time()
    x_train, x_test, x_val  = train_test_validation(pat_slices, pat_df, 0.7,0.2,0.1)
    
    x_train, x_test, x_val  = image_to_np_reshape([x_train, x_test, x_val],pat_df)
    # Reduce footprint by overwriting the array
    pat_slices[:] = None #del pat_slices outside of function
    
    x_train_noisy = noise(x_train)
    x_test_noisy = noise(x_test)
    x_val_noisy = noise(x_test)

    # display([x_train[1][0],x_train_noisy[1][0]])
    x_train, x_test, x_val, x_train_noisy, x_test_noisy, x_val_noisy = expand_dims([x_train, x_test, x_val, x_train_noisy, x_test_noisy, x_val_noisy])

    print("\n"+f"Data augmentation {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

    return x_train, x_test, x_val, x_train_noisy, x_test_noisy, x_val_noisy


def model_building(patients_df: pd.DataFrame, modality: str, x_data, y_data ):
    """
    Given a dataframe of patients, a modality (e.g. "ct"), and the corresponding x and y data, 
    this function returns a trained model for the modality
    
    :param patients_df: The dataframe containing the patient data
    :type patients_df: pd.DataFrame
    :param modality: The modality you want to train the model on
    :type modality: str
    :param x_data: The x-data is the input data for the model. In this case, it's the MRI images
    :param y_data: The target data
    :return: The model
    """
    print(f"{modality} - Model building started".center(50, '_'))
    start_time = time.time()

    shape,idx = patients_df[["dim","tag_idx"]][patients_df.tag.str.contains(modality, case=False)].values[0]
    model = get_model(shape, f"{modality}_model")
    print(x_data[idx].shape)
    #model = train_model(model,x_data[idx], y_data[idx])
    
    print("\n"+f"{modality} - Model building finished {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")


    return model



def main():
    start_time = time.time()
    data_path = "../data/manifest-A3Y4AE4o5818678569166032044/"
    #tags = {"ADC": None,"t2tsetra": (320,320,20)} 
    tags = {"t2tsetra": (320,320,20)} 

    pat_slices, pat_df = preprocess(data_path,tags)
    x_train, x_test, x_val, x_train_noisy, x_test_noisy, x_val_noisy = data_augmentation(pat_slices, pat_df)
    del pat_slices

    t2_model = model_building(pat_df, "t2", x_train_noisy, x_train)
    # t2_model.save("t2_model")

    #test_result = t2_model.evaluate(x_test,y_test, verbose = 1)
    #print("Test accuracy :\t", round (test_result[1], 4))

    # predictions = t2_model.predict(x_test[1])
    # display([x_test[1], predictions])

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
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices('GPU'))
    # print(os.environ["SLURM_PROCID"])
    # print(os.environ["SLURM_NPROCS"])
    main()


