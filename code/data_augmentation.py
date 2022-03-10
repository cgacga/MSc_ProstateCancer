
### Data Augmentation ###

from sklearn.model_selection import train_test_split
from copy import deepcopy
import numpy as np
import pandas as pd
import time
import SimpleITK as sitk
import tensorflow as tf


def train_test_validation(patients_arr, patients_df, train_ratio,test_ratio,validation_ratio):
    """
    The function takes in the image array and dataframe and the split ratio.
    It then splits the array into training, test and validation sets and updates the dataframe.
    
    :param patients_arr: Image array
    :param patients_df: Dataframe with image information
    :param train_ratio: Ratio of training data
    :param test_ratio: Ratio of training data
    :param validation_ratio: Ratio of training data
    :return: training, test and validation sets.
    """

    print(f"\n{'Splitting'.center(50, '.')}")
    # splitting the data into training, test and validation sets
    x_train, x_test, train_df, test_df = train_test_split(patients_arr , patients_df.idx.apply(lambda x: x[0]).unique(), train_size=(1-test_ratio), random_state=42, shuffle=True)
    x_train, x_val, train_df, val_df  = train_test_split(x_train , train_df, train_size=train_ratio/(train_ratio+validation_ratio))

    # Update the dataframe with the new indexes
    df_idx = np.concatenate([train_df, test_df, val_df])
    split_idx = np.concatenate([np.arange(len(i)) for i in [train_df,test_df,val_df]])
    split_names = np.concatenate([["train_df"]*len(train_df), ["test_df"]*len(test_df), ["val_df"]*len(val_df)])
    patients_df[["split","idx"]] = patients_df.idx.apply(lambda x: pd.Series([split_names[df_idx == x[0]][0],(split_idx[df_idx == x[0]][0],x[1])]))
    
    print(f"\n|\tTrain\t|\tTest\t|\tVal\t|")
    print(f"|\t{(len(x_train)/len(patients_arr))*100:.0f}%\t|\t{(len(x_test)/len(patients_arr))*100:.0f}%\t|\t{(len(x_val)/len(patients_arr))*100:.0f}%\t|")
    print(f"|\t{len(x_train)}\t|\t{len(x_test)}\t|\t{len(x_val)}\t|")

    return x_train, x_test, x_val


def image_to_np_reshape(train_test_val_split,patients_df):
    """
    Convert the sitk image to np.array and reshape it to (num_of_slices, height, width, channels)
    
    :param train_test_val_split: The output of the image_to_np_reshape function
    :param patients_df: The dataframe containing the patient information
    :return: a list of 3 np.arrays. Each array contains the images of the patients in the train, test and validation sets.
    """

    print(f"\n{'Converting sitk image to np.array and reshape'.center(50, '.')}")

    start_time = time.time()
    output = []

    # Looping through the train, test, validation sets
    for i,patients_arr in enumerate(train_test_val_split):

        reshaped_arr = np.empty(patients_arr.shape[1],dtype=object)
        for i in range(len(reshaped_arr)):
            # reshaped_arr[i] = np.empty(shape=((patients_arr.shape[0],*patients_arr[0,i].GetSize())),dtype=np.float32)
            dim = patients_arr[0,i].GetSize()
            reshaped_arr[i] = np.empty(shape=((patients_arr.shape[0],dim[2],dim[1],dim[0])),dtype=np.float32)

        for j,pat in enumerate(patients_arr):
            for i,pat_slices in enumerate(pat):
                reshaped_arr[i][j] = sitk.GetArrayFromImage(pat_slices)#.transpose((1,2,0))

        output.append(reshaped_arr)
    
    # Updating the dataframe with new indexes
    patients_df[["tag_idx","pat_idx"]] = patients_df.idx.apply(lambda x: pd.Series([x[1],x[0]]))
    patients_df.drop(columns="idx", inplace=True)

    print(f"\nConversion and reshape finished {(time.time() - start_time):.0f} s")

    #TODO: expand dims here

    return output


def noise(array):
    print(f"\n{'Adding noise'.center(50, '.')}")
    array_deep = deepcopy(array)

    for i,modality in enumerate(array_deep):
        for j,patient in enumerate(modality):
            for k, img_array in enumerate(patient):

                noise_factor = 0.2
                noisy_image = img_array + noise_factor * np.random.normal(
                    loc=0.0, scale=1.0, size=img_array.shape
                )

                array_deep[i][j,k] = np.clip(noisy_image, 0.0, 1.0)
    return array_deep


def expand_dims(array_lst,dim=3):
    """
    Given a list of lists of images, expand the dimensions of the images by 1.
        This is done to make the images compatible with the convolutional layers.
        The function is used to expand the dimensions of the input.
    
    :param array_lst: a list of lists of images
    :return: A list of lists of tensors.
    """
    lst = True
    if not isinstance(array_lst, list):
        array_lst = [array_lst]
        lst = False
    elif len(array_lst)<2:
        lst = False
    for i,array in enumerate(array_lst):
        for j,img_set in enumerate(array):
            # array_lst[i][j] = tf.expand_dims(img_set,-1)
            array_lst[i][j] = tf.repeat(tf.expand_dims(img_set,-1), dim, -1)
    if lst: return array_lst
    else: return array_lst[0]



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

    # Reduce footprint by overwriting the array
    pat_slices[:] = 0 #del pat_slices outside of function

    x_train, x_test, x_val  = image_to_np_reshape([x_train, x_test, x_val],pat_df)
    
    x_train_noisy = noise(x_train)
    x_test_noisy = noise(x_test)
    x_val_noisy = noise(x_val)

    
    # x_train, x_test, x_val, x_train_noisy, x_test_noisy, x_val_noisy = expand_dims([x_train, x_test, x_val, x_train_noisy, x_test_noisy, x_val_noisy])
    x_train, x_test, x_val = expand_dims([x_train, x_test, x_val],dim=1)

    x_train_noisy, x_test_noisy, x_val_noisy = expand_dims([ x_train_noisy, x_test_noisy, x_val_noisy])

    print("\n"+f"Data augmentation {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

    return x_train, x_test, x_val, x_train_noisy, x_test_noisy, x_val_noisy

