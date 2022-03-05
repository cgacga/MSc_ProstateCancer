
### Preprocess ###
import SimpleITK as sitk
import os
import pandas as pd
import time
import numpy as np

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

    print(f"Loading the following:\n{df.groupby(['tag']).size().to_string()}")

    return df.reset_index()


def load_slices(data_path,tags, nslices=False):
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

    if nslices:
        patients_df = patients_df[:10].reset_index()
    
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
    resize_dim = patients_df.groupby(["tag","dim"],as_index=False).idx
    # print(type(resize_dim))
    # print(resize_dim.count())
    # print("ASDASD"*10)
    resize_dim = resize_dim.count().groupby("tag").first().reset_index().rename(columns={"dim":"resize_dim"}).drop(columns=["idx"])

    # print(type(resize_dim))
    # print(resize_dim)
    
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


def preprocess(data_path, tags, nslices = False):
    """
    Loads the slices from the data path, resamples the slices to the desired resolution, normalizes the
    slices and returns the resampled slices and the dataframe

    If size is set to None, then the most common size will be used
        Examples:
            tags = {"ADC":(86,128,20),"t2tsetra": (320,320,20)} 

            tags = {"T2TSETRA": (320,320,20), "adc": None} 

            tags = {"ADC": None}     
    
    :param data_path: path to the folder containing the data
    :param tags: a dictionary of tags and their corresponding sizes
    :return: the preprocessed slices and the dataframe with the metadata.
    """

    print(f"Preprocess started".center(50, '_'))
    start_time = time.time()

    pat_slices, pat_df = load_slices(data_path, tags, nslices=nslices)

    pat_slices, pat_df = resample_pat(pat_slices, pat_df, tags)

    pat_slices = normalize(pat_slices, pat_df)
        
    print("\n"+f"Preprocess finished {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

    return pat_slices, pat_df

