### Importing libraries ###
import matplotlib.pyplot as plt
import SimpleITK as sitk

import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

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
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

# Deprecation removes deprecated warning messages
from tensorflow.python.framework import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = True

# Seed for reproducibility
np.random.seed(42)

### Functions ###

### Img display ###

def display(data):
    """
    Given a list of images, display them in a grid.
    
    :param data: the data to be displayed
    """

    if not isinstance(data, list):
       data = [data]
       img_list = False
    
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
            if not isinstance(img_list, list):
                axarr[j].imshow(data[j, :, :], cmap="gray")    
                axarr[j].axis("off")
            else:
                axarr[j, i].imshow(data[i][j, :, :], cmap="gray")
                axarr[j, i].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()
    
##display([train_data[0,0],train_data[1,0]])
#display([x_train[0][0],x_train[0][1]])

### Preprocess ###

def parse_csv(data_path,type_lst):
    """
    This function takes in a path to the folder where metadata.csv file is, and a list of modalities that we want to extract. 
    The frist folder containing the modality will be selected
    It returns a dataframe with the patients information.
    
    :param data_path: The path to the folder containing the metadata.csv file
    :param type_lst: The list of modalities that you want to include in the dataframe
    :return: a dataframe with the following columns:
        - Series UID: Unique identifier for each series
        - Subject ID: Unique identifier for each patient
        - Series Description: Description of the series
        - Study Date: Date of the study
        - File Location: Location of the file
        - tag: The modality of the series
    """

    if isinstance(type_lst, dict):
        type_lst = type_lst.keys()

    df = pd.read_csv(os.path.normpath(data_path+"metadata.csv"))  

    # Removing subject which gives warning "Non uniform sampling or missing slices detected"
    df.drop(df.index[df["Subject ID"] == "ProstateX-0038"], inplace=True) 

    # Selecting rows which contains the wanted modalities as well as selecting relevant columns
    df = df[df["Series Description"].str.contains("|".join(type_lst), case=False)][["Series UID","Subject ID","Series Description","Study Date","File Location"]]
    
    # Creating a column to make it easier to identify the modality
    df["tag"] = df["Series Description"].str.extract(f"(?i)({'|'.join(type_lst)})", expand=False).fillna(df["Series Description"])

    # Sorting the rows based the folder id
    df = df.assign(helpkey=
            df["File Location"].apply(lambda x: int(os.path.splitext(os.path.basename(x))[0])))\
            .sort_values('helpkey').drop('helpkey',axis = 1)

    # Selecting the first folder
    df = df.groupby(["Subject ID","Study Date","tag"]).first().sort_values(["Subject ID","Study Date","tag"],ascending=[True,False,True])

    # Change the file location to include the whole system path, as well as normalize the path to the correct os path separator
    # If there are problems with loading the files, check if the filepath is correct within the dataframe compared the file path in the os.
    df["File Location"] = df["File Location"].apply(lambda x: os.path.normpath(data_path+x)) #abspath   

    print(f"The series contains following number of each type:\n{df.groupby(['tag']).size().to_string()}")

    return df.reset_index()


def load_slices(data_path,tags):
    """
    Given a path to the data, it loads the slices and stores them in a numpy array, if the Series Description contains the dictionary keys
    
    :param data_path: Path to the data folder
    :param tags: Dictionary or list of tags that you want to read from the dicom file (Case insensitive)
    :return: 
        patients_arr: Array of images of shape (number of patients, number of modalities)
        patients_df: Dataframe containing the information of the series
    """

    print("\n"+f"Loading slices".center(50, '.'))
    start_time = time.time()

    patients_df = parse_csv(data_path,tags)

    #patients_df = patients_df[:10].reset_index()
    
    # Assigning empty array with shape = (number of patients,number of modalities), in our case (346,2)
    patients_arr = np.empty(shape=(np.ceil(len(patients_df)/len(patients_df["tag"].unique())).astype(int),len(patients_df["tag"].unique())),dtype=object)
    
    # Array to hold the id of each patient (done here and not in parse_csv to be sure that the id is assigned correctly)
    idx = np.empty(shape=(len(patients_df)),dtype=tuple)
    reader = sitk.ImageSeriesReader()
    i = 0
    
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
    
    print(f"Loading finished {(time.time() - start_time):.0f} s")
    return patients_arr, patients_df


def getsize(patients_arr,patients_df, force_dim={}):
    """
    Gather the size of each image and select the resize dimension based on the most common dimension if no dimension is specified
    
    :param patients_arr: The array of images
    :param patients_df: The dataframe containing the patients information
    :param force_dim: Dictonary containing type and dimension
    :return: Dataframe with updated image dimensions
    """

    # Gather the size of each image
    dim_arr = []
    for _, pat in patients_df.iterrows():
            dim_arr.append(patients_arr[pat.idx].GetSize())
    patients_df["dim"] = dim_arr
        
    # Selecting the resize dimension based on the most common dimension
    resize_dim = patients_df.groupby(["tag"],as_index=False)["dim"].value_counts().groupby("tag").first().reset_index().rename(columns={"dim":"resize_dim"}).drop(columns=["count"])
    # if the user wants to select their own dimension for all or just a specific modality
    force_dim = {modality: dim for modality, dim in force_dim.items() if dim}
    if force_dim:
        resize_dim.loc[resize_dim.tag.isin(force_dim.keys()), 'resize_dim'] = resize_dim.tag.map(force_dim)
    patients_df = patients_df.merge(resize_dim, on="tag", how="left", suffixes=('', '_duplicate')).filter(regex='^(?!.*_duplicate)')

    print(f"\nCurrent resolution in the series pr modality:\n{patients_df.groupby(['tag']).dim.value_counts().to_frame('count')}")   
    if not (patients_df.dim.equals(patients_df.resize_dim)):
        print(f"\nChosen resampling dimension of each modality:\n{resize_dim.to_string(index=False)}")

    return patients_df


def resample_pat(patients_arr,patients_df, force_dim={}, interpolator=sitk.sitkLinear, default_value=0):
    """
    Resample the images to the given resolution
    
    :param patients_arr: The array of images to resample
    :param patients_df: The dataframe containing the patients information
    :param force_dim: Dictonary containing type and dimension
    :param interpolator: The interpolator to use for resampling
    :param default_value: The value to assign to all voxels in the output image, defaults to 0
    :return: the resampled images and the dataframe with the new dimensions.
    """

    print("\n"+f"Resampling".center(50, '.'))
    start_time = time.time()

    print("\n"+f"~Selecting resolution for resampling~".center(50, ' '))
    patients_df = getsize(patients_arr,patients_df, force_dim)
    print("\n"+f"~Resamplig started~".center(50, ' '))

    new_spacing = np.zeros(3)
    filter = sitk.ResampleImageFilter()

    for _, pat in patients_df.iterrows():
        pat_slices = patients_arr[pat.idx]
        old_size = pat_slices.GetSize()
        new_size = pat.resize_dim
        if  old_size == new_size: continue # Skip resize of images with correct dimensions
        old_spacing = pat_slices.GetSpacing()
        new_spacing = [old_spacing[i] / new_dim * old_size[i] for i,new_dim in enumerate(new_size)]
        
        filter.SetOutputSpacing(new_spacing)
        filter.SetInterpolator(interpolator)
        filter.SetOutputOrigin(pat_slices.GetOrigin())
        filter.SetOutputDirection(pat_slices.GetDirection())
        filter.SetSize(new_size)
        filter.SetDefaultPixelValue(default_value)
        patients_arr[pat.idx] = filter.Execute(pat_slices)

    # Check the images dimensions
    print("\n"+f"~Resampling ended, checking new resolution~".center(50, ' '))
    patients_df = getsize(patients_arr,patients_df)
    patients_df.drop(columns=["resize_dim"],inplace=True)
    
    print(f"\nResampling finished {(time.time() - start_time):.0f} s")
    return  patients_arr, patients_df


def normalize(patients_arr,patients_df):
    """
    Normalize the image set
    
    :param patients_arr: The array of images to be normalized
    :param patients_df: The dataframe containing the patient information
    :return: the normalized images.
    """

    print("\n"+f"Normalization".center(50, '.'))
    start_time = time.time()

    minmax_filter = sitk.MinimumMaximumImageFilter()
    t2_upper_perc, t2_lower_perc = [], []
    adc_max, adc_min = 0, np.iinfo(np.int32).max
     
    # Iterate over the image set to gather the intensity values
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

    # Iterate over the set to normalize the values
    for _, pat in patients_df.iterrows():
        if "t2" in pat.tag:
            normalization_filter.SetWindowMaximum(t2_upper)
            normalization_filter.SetWindowMinimum(t2_lower)
        elif "ADC" in pat.tag:
            normalization_filter.SetWindowMaximum(adc_max) #400
            normalization_filter.SetWindowMinimum(adc_min) #-1000
    
        float_series = cast_image_filter.Execute(patients_arr[pat.idx])
        patients_arr[pat.idx] = normalization_filter.Execute(float_series)

    print(f"\nNormalization finished {(time.time() - start_time):.0f} s")
    return patients_arr


### Data Augmentation ###

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
            reshaped_arr[i] = np.empty(shape=((patients_arr.shape[0],*patients_arr[0,i].GetSize())),dtype=np.float32)

        for j,pat in enumerate(patients_arr):
            for i,pat_slices in enumerate(pat):
                reshaped_arr[i][j] = sitk.GetArrayFromImage(pat_slices).transpose((2,1,0))

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

                noise_factor = 0.1
                noisy_image = img_array + noise_factor * np.random.normal(
                    loc=0.0, scale=1.0, size=img_array.shape
                )

                array_deep[i][j,k] = np.clip(noisy_image, 0.0, 1.0)
    return array_deep


def expand_dims(array_lst):
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
            array_lst[i][j] = tf.expand_dims(img_set,axis=4)
    if lst: return array_lst
    else: return array_lst[0]


### Model building ###


def get_model(dim, name="autoencoder"):
    """
    It creates a model that accepts a 3D input of shape (width, height, depth, 1) and returns a 3D
    output of the same shape.
    
    :param dim: The shape of the input data
    :param name: The Model's name
    :return: The autoencoder model
    """
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


def train_model(model, x_data, y_data):
    """
    Function to train the model.
    
    :param model: The model to train
    :param x_data: The training data
    :param y_data: The labels for the training data
    :return: The trained model.
    """
    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(f"{model.name} - 3d_image_classification.h5", save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

    model.fit(
        x=x_data,
        y=y_data,
        epochs=50,
        batch_size=128,
        shuffle=True,
        verbose=1,
        validation_data=(x_data, y_data),
        callbacks=[checkpoint_cb, early_stopping_cb],
    )

    print("Train accuracy :\t", round (model.history['acc'][0], 4))

    return model

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
    # adc_shape,adc_idx = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains("adc", case=False)].values[0]
    # adc_model = get_model(adc_shape, "adc_model")
    # adc_model = train_model(adc_model,x_data[adc_idx], x_data )

    # t2_shape,t2_idx = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains("t2tsetra", case=False)].values[0]
    # t2_model = get_model(t2_shape, "t2_model")
    # t2_model = train_model(t2_model,x_data[t2_idx], x_data )



    shape,idx = patients_df[["dim","tag_idx"]][patients_df.tag.str.contains(modality, case=False)].values[0]
    model = get_model(shape, f"{modality}_model")
    model = train_model(model,x_data[idx], y_data)

    return model


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






def main():
    data_path = "../Data/manifest-A3Y4AE4o5818678569166032044/"
    tags = {"ADC": None,"t2tsetra": (320,320,20)} 

    pat_slices, pat_df = preprocess(data_path,tags)
    x_train, x_test, x_val, x_train_noisy, x_test_noisy, x_val_noisy = data_augmentation(pat_slices, pat_df)
    del pat_slices

    t2_model = model_building(pat_df, "t2", x_train_noisy, x_train)
    t2_model.save("t2_model")

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

    return pat_df, x_train, x_test, x_val, x_train_noisy, x_test_noisy


# df, x_train, x_test, x_val, x_train_noisy, x_test_noisy = main()

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices('GPU'))
    main()

