
### Data Augmentation ###

from copy import deepcopy
import numpy as np
import pandas as pd
import time
import SimpleITK as sitk



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
    y_train, y_test, y_val  = train_test_validation(pat_slices, pat_df, 0.7,0.2,0.1)

    # Reduce footprint by overwriting the array
    pat_slices[:] = 0 #del pat_slices outside of function

    # x_train, x_test, x_val = augmentation([y_train, y_test, y_val],pat_df)

    # y_train, y_test, y_val  = image_to_np_reshape([y_train, y_test, y_val],pat_df)

    # x_train, x_test, x_val  = image_to_np_reshape([x_train, x_test, x_val],pat_df)


    #TODO: print the plots to check

    
    
    # x_train_noisy = noise(x_train)
    # x_test_noisy = noise(x_test)
    # x_val_noisy = noise(x_val)

    #TODO: patch pictures (try to do this onlie)
        # more than 3 patches at least (maybe use this as a variable?)
    #TODO: rotate patches

    
    # x_train, x_test, x_val, x_train_noisy, x_test_noisy, x_val_noisy = expand_dims([x_train, x_test, x_val, x_train_noisy, x_test_noisy, x_val_noisy])
    # x_train, x_test, x_val = expand_dims([x_train, x_test, x_val],dim=1)

    # x_train_noisy, x_test_noisy, x_val_noisy = expand_dims([ x_train_noisy, x_test_noisy, x_val_noisy],dim=1)

    print("\n"+f"Data augmentation {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

    #x_train = agumentated data
    #y_train = original data
    

    return x_train, x_test, x_val, y_train, y_test, y_val

