
### Data Augmentation ###

from copy import deepcopy
import numpy as np
import pandas as pd
import time
import SimpleITK as sitk



# def noise(array):
#     print(f"\n{'Adding noise'.center(50, '.')}")
#     array_deep = deepcopy(array)

#     for i,modality in enumerate(array_deep):
#         for j,patient in enumerate(modality):
#             for k, img_array in enumerate(patient):

#                 noise_factor = 0.2
#                 noisy_image = img_array + noise_factor * np.random.normal(
#                     loc=0.0, scale=1.0, size=img_array.shape
#                 )

#                 array_deep[i][j,k] = np.clip(noisy_image, 0.0, 1.0)
#     return array_deep



# def augmentation(train_test_val_split,patients_df):

#     print(f"\n{'Data augmentation'.center(50, '.')}")

#     start_time = time.time()
#     output = []

#     # Looping through the train, test, validation sets
#     for i,patients_arr in enumerate(train_test_val_split):
#         for j,pat in enumerate(patients_arr):
#             for k,pat_slices in enumerate(pat):
#                 reshaped_arr[k][j] = sitk.GetArrayFromImage(pat_slices)#.transpose((1,2,0))

#         output.append(reshaped_arr)
    
#     # Updating the dataframe with new indexes
#     patients_df[["tag_idx","pat_idx"]] = patients_df.idx.apply(lambda x: pd.Series([x[1],x[0]]))
#     patients_df.drop(columns="idx", inplace=True)

#     print(f"\nConversion and reshape finished {(time.time() - start_time):.0f} s")

#     #TODO: expand dims here... or not?

#     return output



# def data_augmentation(pat_slices, pat_df):
#     """
#     This function takes in the patient slices and the patient dataframe and returns the train, test and
#     validation data
    
#     :param pat_slices: The list of slices that we have extracted from the patients
#     :param pat_df: The dataframe containing the patient id's of the slices
#     :return: the training, test and validation data sets.
#     """

#     print(f"Data augmentation started".center(50, '_'))
#     start_time = time.time()
#     y_train, y_test, y_val  = train_test_validation(pat_slices, pat_df, 0.7,0.2,0.1)

#     # Reduce footprint by overwriting the array
#     pat_slices[:] = 0 #del pat_slices outside of function

#     # x_train, x_test, x_val = augmentation([y_train, y_test, y_val],pat_df)

#     # y_train, y_test, y_val  = image_to_np_reshape([y_train, y_test, y_val],pat_df)

#     # x_train, x_test, x_val  = image_to_np_reshape([x_train, x_test, x_val],pat_df)


#     #TODO: print the plots to check

    
    
#     # x_train_noisy = noise(x_train)
#     # x_test_noisy = noise(x_test)
#     # x_val_noisy = noise(x_val)

#     #TODO: patch pictures (try to do this onlie)
#         # more than 3 patches at least (maybe use this as a variable?)
#     #TODO: rotate patches

    
#     # x_train, x_test, x_val, x_train_noisy, x_test_noisy, x_val_noisy = expand_dims([x_train, x_test, x_val, x_train_noisy, x_test_noisy, x_val_noisy])
#     # x_train, x_test, x_val = expand_dims([x_train, x_test, x_val],dim=1)

#     # x_train_noisy, x_test_noisy, x_val_noisy = expand_dims([ x_train_noisy, x_test_noisy, x_val_noisy],dim=1)

#     print("\n"+f"Data augmentation {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

#     #x_train = agumentated data
#     #y_train = original data
    

#     return x_train, x_test, x_val, y_train, y_test, y_val

