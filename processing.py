# %%

from attr import asdict
from matplotlib import image
import matplotlib.pyplot as plt
import SimpleITK as sitk

import os
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from urllib3 import add_stderr_logger

from copy import copy, deepcopy

np.random.seed(42)

# %%

### img display ###

def display(data):

    img = data.copy()
    if not isinstance(data, list):
       data = [data]
    
    columns_data = len(data)
    rows_data = max([data[i].shape[0] for i in range(columns_data)])

    # width_px = max([data[i].shape[1] for i in range(columns_data)])
    # height_px = max([data[i].shape[2] for i in range(columns_data)])

    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(10,60), #tune this 
    )
    for i in range(columns_data):
        for j in range(rows_data):
            if not isinstance(img, list):
                axarr[j].imshow(img[j, :, :], cmap="gray")    
                axarr[j].axis("off")
            else:
                axarr[j, i].imshow(img[i][j, :, :], cmap="gray")
                axarr[j, i].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()
    
##display([train_data[0,0],train_data[1,0]])
#display([x_train[0][0],x_train[0][1]])

### Preprocess ###

def parse_csv(data_path,type_lst):
    '''
    This function takes in a path to the folder where metadata.csv file is, and a list of strings that you want to search for. 
    It returns a dataframe with the patients file information.
        
    :param data_path: the path to the data directory
    :param type_lst: list of strings that are the series descriptions you want to include in the dataset
    :return: a dataframe with the following columns:
        - Series UID
        - Subject ID
        - Series Description
        - Study Date
        - File Location
        - tag
        - idx
    '''
    
    df = pd.read_csv(os.path.normpath(data_path+"metadata.csv"))  

    # Removing subject which gives warning
    #df.drop(df.index[df["Subject ID"] == "ProstateX-0038"], inplace=True) 

    # Selecting rows which contains the wanted modalities as well as selecting relevant columns
    df = df[df["Series Description"].str.contains("|".join(type_lst))][["Series UID","Subject ID","Series Description","Study Date","File Location"]]

    # Creating a column to make it easier to identify the modality
    df["tag"] = df["Series Description"].str.extract(f"({'|'.join(type_lst)})", expand=False).fillna(df["Series Description"])

    # Sorting the rows based the folder id
    df = df.assign(helpkey=
            df["File Location"].apply(lambda x: int(os.path.splitext(os.path.basename(x))[0])))\
            .sort_values('helpkey')#.drop('helpkey',axis = 1)

    # Selecting the first folder
    df = df.groupby(["Subject ID","Study Date","tag"]).first().sort_values(["Subject ID","Study Date","tag"],ascending=[True,False,True])

    # Change the file location to include the whole system path, as well as normalize the path to the correct os path separator
    # If there are problems with loading the files, check if the filepath is correct within the dataframe compared the file path in the os.
    df["File Location"] = df["File Location"].apply(lambda x: os.path.normpath(data_path+x)) #abspath

    print(f"The series contains following number of each type:\n{df.groupby(['tag']).size().to_string()}")

    return df.reset_index()



def load_slices(patients_df):
    
    # Assigning empty array with shape = (number of patients,number of modalities), in our case (346,2)
    patients_arr = np.empty(shape=(np.ceil(len(patients_df)/len(patients_df["tag"].unique())).astype(int),len(patients_df["tag"].unique())),dtype=object)
    # Array to hold the id of each patient (done here and not in parse_csv to be sure that the id is assigned correctly)
    idx = np.empty(shape=(len(patients_df)),dtype=tuple)
    i = 0
    reader = sitk.ImageSeriesReader()
    
    for _, group in patients_df.groupby(["Subject ID","Study Date"]):
        j = 0
        for row, patient in group.iterrows():
            # Building the image array
            # Includes UID in case of multiple dicom sets in the folder & sorting in ascending order
            reader.SetFileNames(reader.GetGDCMSeriesFileNames(directory = patient["File Location"], seriesID = patient["Series UID"])[::-1]) 
            patients_arr[i,j] = reader.Execute()
            
            idx[row] = (i,j)
            j = j+1
        i = i+1

    # Store the index of the patient
    patients_df["idx"] = idx
    
    return patients_arr


def getsize(patients_arr,patients_df,include_depth = True, force_dim={}):

    dim_arr = []
    for _, pat in patients_df.iterrows():
            # Gather the size of each image
            dim_arr.append(patients_arr[pat.idx].GetSize())

    if include_depth:
        patients_df["dim"] = dim_arr
    else:
        dim_arr = np.array(dim_arr) 
        patients_df = patients_df.join(pd.DataFrame({"dim":zip(dim_arr[:,0],dim_arr[:,1]),"z":dim_arr[:,2]}, index=patients_df.index), lsuffix=('', '_duplicate')).filter(regex='^(?!.*_duplicate)')
    #TODO: concatinate z column with dim
        
    resize_dim = patients_df.groupby(["tag"],as_index=False)["dim"].value_counts().groupby("tag").first().reset_index().rename(columns={"dim":"resize_dim"}).drop(columns=["count"])

    if force_dim:
        resize_dim.loc[resize_dim.tag.isin(force_dim.keys()), 'resize_dim'] = resize_dim.tag.map(force_dim)

    patients_df = patients_df.merge(resize_dim, on="tag", how="left", suffixes=('', '_duplicate')).filter(regex='^(?!.*_duplicate)')

    print(f"\nCurrent resolution in the series pr modality:\n{patients_df.groupby(['tag']).dim.value_counts().to_frame('count')}")   

    if not (patients_df.dim.equals(patients_df.resize_dim)):
        print(f"\nChosen resampling dimension of each modality:\n{resize_dim.to_string(index=False)}")

    return patients_df


def resample_pat(patients_arr,patients_df, include_depth = True, force_dim={}, interpolator=sitk.sitkLinear, default_value=0):

    new_spacing = np.zeros(3)

    print("\n"+f"~Selecting resolution for resampling~".center(50, '.'))
    patients_df = getsize(patients_arr,patients_df, include_depth, force_dim)
    print("\n"+f"~Resamplig started~".center(50, '.'))
    for _, pat in patients_df.iterrows():
        pat_slices = patients_arr[pat.idx]
        old_size = pat_slices.GetSize()
        ###if  old_size == pat.resize_dim: continue
        dim = pat.resize_dim[::-1]
        new_width, new_height = dim[0],dim[1]
        old_spacing = pat_slices.GetSpacing()
        new_spacing[0] = old_spacing[0] / new_width * old_size[0]
        new_spacing[1] = old_spacing[1] / new_height * old_size[1]
        if include_depth and len(dim)>2: 
            new_depth = dim[2]
            new_spacing[2] = old_spacing[2] / new_depth * old_size[2]
            ##new_depth = old_spacing[2] / new_depth * old_size[2]
        else:
            # if we assume number of slices will be the same one
            new_spacing[2] = old_spacing[2]
            new_depth = old_spacing[2] / new_spacing[2] * old_size[2]
        new_size = [int(new_width), int(new_height), int(new_depth)]
        
        filter = sitk.ResampleImageFilter()
        filter.SetOutputSpacing(new_spacing)
        filter.SetInterpolator(interpolator)
        filter.SetOutputOrigin(pat_slices.GetOrigin())
        filter.SetOutputDirection(pat_slices.GetDirection())
        filter.SetSize(new_size)
        filter.SetDefaultPixelValue(default_value)
        patients_arr[pat.idx] = filter.Execute(pat_slices)

    print("\n"+f"~Resampling ended, checking new resolution~".center(50, '.'))
    patients_df = getsize(patients_arr,patients_df,include_depth)
    patients_df.drop(columns=["resize_dim"],inplace=True)

    return  patients_arr , patients_df


def normalize(patients_arr,patients_df):

    minmax_filter = sitk.MinimumMaximumImageFilter()
    t2_upper_perc, t2_lower_perc = [], []
    adc_max = 0
    adc_min = np.iinfo(np.int32).max

    for _, pat in patients_df.iterrows():
        if "t2" in pat.tag:
            t2_array = sitk.GetArrayViewFromImage(patients_arr[pat.idx])
            for img in t2_array:
                t2_upper_perc.append(np.percentile(img, 99))
                t2_lower_perc.append(np.percentile(img, 1))
        elif "ADC" in pat.tag:
            minmax_filter.Execute(patients_arr[pat.idx])
            adc_max_intensity = minmax_filter.GetMaximum()
            adc_min_intensity = minmax_filter.GetMinimum()
            if adc_max_intensity > adc_max: adc_max = adc_max_intensity
            if adc_min_intensity < adc_min: adc_min = adc_min_intensity

    cast_image_filter = sitk.CastImageFilter()
    cast_image_filter.SetOutputPixelType(sitk.sitkFloat32)
    normalization_filter = sitk.IntensityWindowingImageFilter()        
    normalization_filter.SetOutputMaximum(1.0)
    normalization_filter.SetOutputMinimum(0.0)
    if (t2_upper_perc and t2_lower_perc):
        t2_upper = np.percentile(t2_upper_perc, 99)
        t2_lower = np.percentile(t2_lower_perc, 1)

    for _, pat in patients_df.iterrows():
        if "t2" in pat.tag:
            normalization_filter.SetWindowMaximum(t2_upper)
            normalization_filter.SetWindowMinimum(t2_lower)
        elif "ADC" in pat.tag:
            normalization_filter.SetWindowMaximum(adc_max) #400
            normalization_filter.SetWindowMinimum(adc_min) #-1000
    
        float_series = cast_image_filter.Execute(patients_arr[pat.idx])
        patients_arr[pat.idx] = normalization_filter.Execute(float_series)

    return patients_arr




### Data Augmentation ###

def train_test_validation(patients_arr, patients_df, train_ratio,test_ratio,validation_ratio):

    # splitting the data into training, test and validation sets

    # x_train, x_test, train_df, test_df = train_test_split(patients_arr , patients_df.idx.unique(), train_size=(1-test_ratio), random_state=101, shuffle=True)

    x_train, x_test, train_df, test_df = train_test_split(patients_arr , patients_df.idx.apply(lambda x: x[0]).unique(), train_size=(1-test_ratio), random_state=101, shuffle=True)

    x_train, x_val, train_df, val_df  = train_test_split(x_train , train_df, train_size=train_ratio/(train_ratio+validation_ratio))

    df_idx = np.concatenate([train_df, test_df, val_df])
    split_idx = np.concatenate([np.arange(len(i)) for i in [train_df,test_df,val_df]])
    split_names = np.concatenate([["train_df"]*len(train_df), ["test_df"]*len(test_df), ["val_df"]*len(val_df)])

    #patients_df[["split","idx"]] = patients_df.idx.apply(lambda x: pd.Series([split_names[df_idx == x][0],split_idx[df_idx == x][0]]))
    patients_df[["split","idx"]] = patients_df.idx.apply(lambda x: pd.Series([split_names[df_idx == x[0]][0],(split_idx[df_idx == x[0]][0],x[1])]))
    
    print(f"\n|\tTrain\t|\tTest\t|\tVal\t|")
    print(f"|\t{(len(x_train)/len(patients_arr))*100:.0f}%\t|\t{(len(x_test)/len(patients_arr))*100:.0f}%\t|\t{(len(x_val)/len(patients_arr))*100:.0f}%\t|")
    print(f"|\t{len(x_train)}\t|\t{len(x_test)}\t|\t{len(x_val)}\t|")

    return x_train, x_test, x_val

def image_to_np_reshape(train_test_val_split,patients_df):

    output = []

    for i,patients_arr in enumerate(train_test_val_split):

        reshaped_arr = np.empty(patients_arr.shape[1],dtype=object)
        for i in range(len(reshaped_arr)):
            img_size = patients_arr[0,i].GetSize()
            reshaped_arr[i] = np.empty(shape=((patients_arr.shape[0],*img_size[::-1])),dtype=np.float32)

        # for _, pat in patients_df.iterrows():
        #     reshaped_arr[pat.idx[1]][pat.idx[0]] = sitk.GetArrayFromImage(pat_slices[pat.idx])
            #pat.idx = pat.idx[:-1]

        for j,pat in enumerate(patients_arr):
            for i,pat_slices in enumerate(pat):
                reshaped_arr[i][j] = sitk.GetArrayFromImage(pat_slices)
        
        output.append(reshaped_arr)
    
    patients_df[["tag_idx","pat_idx"]] = patients_df.idx.apply(lambda x: pd.Series([x[1],x[0]]))
    patients_df.drop(columns="idx", inplace=True)

    print(f"~reshape, new shape .,.,.,. ~".center(50, ' '))
    return output


def noise(array):
    array_deep = deepcopy(array)

    for i,modality in enumerate(array_deep):
        for j,patient in enumerate(modality):
            for k, img_array in enumerate(patient):

                noise_factor = 0.1
                noisy_image = img_array + noise_factor * np.random.normal(
                    loc=0.0, scale=1.0, size=img_array.shape
                )

                array_deep[i][j,k] = np.clip(noisy_image, 0.0, 1.0)
    return array_deep


def expand_dims(array):
    for i,modality in enumerate(array):
        array[i] = tf.expand_dims(modality,axis=4)
    return array


### Main functions ###

def preprocess():

    data_path = "../Data/manifest-A3Y4AE4o5818678569166032044/"
    tags = ["t2tsetra","ADC"]
    #force_dim = {"ADC":(12,123,321),"t2tsetra": (20,320,320)} 
    force_dim = {"t2tsetra": (20,320,320)} 

    start_preprocess = time.time()

    pat_df = parse_csv(data_path,tags)

    pat_df = pat_df[78:80].reset_index()

    print("\n"+f"Loading slices".center(50, '.'))
    start = time.time()
    pat_slices = load_slices(pat_df)
    print(f"Loading finished {(time.time() - start):.1f} s")
    
    print("\n"+f"Resampling".center(50, '.'))
    start = time.time()
    pat_slices, pat_df = resample_pat(pat_slices,pat_df,include_depth=True, force_dim = force_dim)
    print(f"\nResampling finished {(time.time() - start):.1f} s")

    print("\n"+f"Normalization".center(50, '.'))
    start = time.time()
    pat_slices = normalize(pat_slices,pat_df)
    print(f"\nNormalization finished {(time.time() - start):.1f} s")
        
    print(f"\nPreprocess finished {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_preprocess))}")

    return pat_slices, pat_df


##%%

def data_augmentation(pat_slices, pat_df):


    print(f"\n{'Splitting'.center(50, '.')}")
    x_train, x_test, x_val  = train_test_validation(pat_slices, pat_df, 0.7,0.2,0.1)

    print(f"\n{'Converting sitk image to np.array'.center(50, '.')}")
    x_train, x_test, x_val  = image_to_np_reshape([x_train, x_test, x_val],pat_df)

    print(f"\n{'Adding noise'.center(50, '.')}")
    x_train_noisy = noise(x_train)
    x_test_noisy = noise(x_test)

    display([x_train[1][0],x_train_noisy[1][0]])

    x_train, x_test, x_val, x_train_noisy, x_test_noisy = [expand_dims(array) for array in (x_train, x_test, x_val, x_train_noisy, x_test_noisy)]

    return x_train, x_test, x_val, x_train_noisy, x_test_noisy

def model_building():

    return



def main():
    pat_slices, pat_df = preprocess()
    x_train, x_test, x_val, x_train_noisy, x_test_noisy = data_augmentation(pat_slices, pat_df)

    return


pat_slices, pat_df = preprocess()
x_train, x_test, x_val, x_train_noisy, x_test_noisy = data_augmentation(pat_slices, pat_df)



#%%

data_path = "../Data/manifest-A3Y4AE4o5818678569166032044/"
tags = ["t2tsetra","ADC"]
#force_dim = {"ADC":(12,123,321),"t2tsetra": (20,320,320)} 
#force_dim = {"t2tsetra": (20,320,320)} 



pat_df = parse_csv(data_path,tags)

# pat_df = pat_df[400:460].reset_index()

pat_slices = load_slices(pat_df)

pat_slices, pat_df = resample_pat(pat_slices,pat_df,include_depth=True)#, force_dim = force_dim)

#%%

print(x_train.shape)
print(x_train[0].shape)
print(x_train[1].shape)

# %%

def get_model(dim, name="autoencoder"):
    # shape = (width, height, depth, 1)
    inputs = keras.Input(shape=(dim[1], dim[2], dim[3], 1))

    # Encoder
    x = layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling3D((2, 2, 2), padding="same")(x)
    x = layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(x)
    encoded = layers.MaxPooling3D((2, 2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv3DTranspose(32, (3, 3, 3), strides=2, activation="relu", padding="same")(encoded)
    x = layers.Conv3DTranspose(32, (3, 3, 3), strides=2, activation="relu", padding="same")(x)
    decoded = layers.Conv3D(1, (3, 3, 3), activation="sigmoid", padding="same")(x)

    # Autoencoder
    autoencoder = Model(inputs, decoded, name=name)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.summary()
    return autoencoder


# pat_df.dim[pat_df.tag.str.contains("ADC")][0]
adc_model = get_model(x_train[0].shape, "adc_model")
t2_model = get_model(x_train[1].shape, "t2_model")


# %%

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)


qwe_train = x_train[1]
qwe_test  = x_test[1]

t2_model.fit(
    x=qwe_train,
    y=qwe_train,
    epochs=50,
    batch_size=128,
    shuffle=True,
    verbose=1,
    validation_data=(qwe_test, qwe_test),
    callbacks=[checkpoint_cb, early_stopping_cb],
)


# %%

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)


qwe_train = x_train[0]
qwe_test  = x_test[0]

adc_model.fit(
    x=qwe_train,
    y=qwe_train,
    epochs=50,
    batch_size=128,
    shuffle=True,
    verbose=1,
    validation_data=(qwe_test, qwe_test),
    callbacks=[checkpoint_cb, early_stopping_cb],
)

# %%


predictions = adc_model.predict(x_test[0])
display([x_test[0], predictions])

predictions = t2_model.predict(x_test[1])
display([x_test[1], predictions])


# %%


# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)


qwe_train = x_train[1]
qwe_test  = x_test[1]
qwe_train_noisy = x_train_noisy[1]
qwe_test_noisy = x_test_noisy[1]

t2_model.fit(
    x=qwe_train_noisy,
    y=qwe_train,
    epochs=100,
    batch_size=128,
    shuffle=True,
    verbose=1,
    validation_data=(qwe_test_noisy, qwe_test),
    callbacks=[checkpoint_cb, early_stopping_cb],
)


# %%

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)


qwe_train = x_train[0]
qwe_test  = x_test[0]

adc_model.fit(
    x=qwe_train_noisy,
    y=qwe_train,
    epochs=100,
    batch_size=128,
    shuffle=True,
    verbose=1,
    validation_data=(qwe_test_noisy, qwe_test),
    callbacks=[checkpoint_cb, early_stopping_cb],
)
# %%

predictions = adc_model.predict(x_test[0])
display([x_test[0], predictions])

predictions = t2_model.predict(x_test[1])
display([x_test[1], predictions])

