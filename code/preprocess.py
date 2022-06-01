
### Preprocess ###
import os, time
import numpy as np
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from params import modality
from data_augmentation import set_seed


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
    
    findings = pd.read_csv(os.path.normpath(data_path+"ProstateX-Findings-Train.csv"))
    findings = findings[["ProxID","ClinSig"]]
    findings.rename(columns={'ProxID': 'Subject ID'}, inplace=True)
    findings = findings.sort_values("ClinSig")[::-1].drop_duplicates(['Subject ID'], keep='first')
    findings = findings.sort_values("Subject ID").reset_index(drop=True)
    findings["ClinSig"].mask(findings["ClinSig"], "significant", inplace=True)
    findings["ClinSig"].mask(findings["ClinSig"] == False, "non-significant", inplace=True)
    
    # Removing subject which gives warning "Non uniform sampling or missing slices detected"
    df.drop(df.index[df["Subject ID"] == "ProstateX-0038"], inplace=True) 


    # Selecting rows which contains the wanted modalities as well as selecting relevant columns
    df = df[df["Series Description"].str.contains("|".join(type_lst), case=False)][["Series UID","Subject ID","Series Description","Study Date","File Location"]]

    df = df.merge(findings, how="left", on="Subject ID")
    
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
        patients_df = patients_df[:30].reset_index()
    
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


    #TODO: change to minimum 32 in depth (z)
    

    # Gather the size of each image
    dim_arr = []
    for _, pat in patients_df.iterrows():
            dim = patients_arr[pat.idx].GetSize()
            #dim_arr.append((dim[1],dim[0],dim[2]))
            dim_arr.append((dim[2],dim[1],dim[0]))
            
                #patients_arr[pat.idx].GetSize().transpose((1,0,2)))
    patients_df["dim"] = dim_arr
        
    # Selecting the resize dimension based on the most common dimension
    resize_dim = patients_df.groupby(["tag","dim"],as_index=False).idx
    resize_dim = resize_dim.count().groupby("tag").first().reset_index().rename(columns={"dim":"resize_dim"}).drop(columns=["idx"])

    # if the user wants to select their own dimension for all or just a specific modality
    #force_dim = {modality_name.lower(): (dim if dim else (*resize_dim.resize_dim[resize_dim.tag.str.contains(modality_name, case=False)][0][:-1],32)) for modality_name, dim in force_dim.items() }
    
    force_dim = {modality_name.lower(): (dim if dim else (32,*resize_dim.resize_dim[resize_dim.tag.str.contains(modality_name, case=False)].item()[1:])) for modality_name, dim in force_dim.items() }
    if force_dim:
        resize_dim.loc[resize_dim.tag.str.lower().isin(force_dim.keys()), 'resize_dim'] = resize_dim.tag.str.lower().map(force_dim)
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

    #TODO: save before and after image to show difference in the report. Due to resampling and antialiasing issues
    #https://simpleitk.org/SPIE2018_COURSE/images_and_resampling.pdf
    #https://stackoverflow.com/questions/48065117/simpleitk-resize-images


    for _, pat in patients_df.iterrows():
        pat_slices = patients_arr[pat.idx]
        old_size = pat_slices.GetSize()
        new_size = (pat.resize_dim[2],pat.resize_dim[1],pat.resize_dim[0])
        #new_size = (pat.resize_dim[1],pat.resize_dim[0],pat.resize_dim[2])
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
    #normalization_filter.SetOutputMaximum(255)
    #normalization_filter.SetOutputMinimum(0)
    if (t2_upper_perc and t2_lower_perc):
        t2_upper = np.percentile(t2_upper_perc, 99)
        t2_lower = np.percentile(t2_lower_perc, 1)

    # Iterate over the set to normalize the values
    for _, pat in patients_df.iterrows():
        if "t2" in pat.tag:
            normalization_filter.SetWindowMaximum(t2_upper)
            normalization_filter.SetWindowMinimum(t2_lower)
        elif "ADC" in pat.tag:
            normalization_filter.SetWindowMaximum(adc_max) 
            normalization_filter.SetWindowMinimum(adc_min) 
    
        float_series = cast_image_filter.Execute(patients_arr[pat.idx])
        patients_arr[pat.idx] = normalization_filter.Execute(float_series)

    print(f"\nNormalization finished {(time.time() - start_time):.0f} s")
    return patients_arr


def train_test_validation(patients_array, patients_dataframe, ratio):
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
    old_pat_index = patients_dataframe.idx.apply(lambda x: x[0]).unique()
    patients_df=patients_dataframe.copy()
    patients_df=patients_df[~patients_df['ClinSig'].isna()].copy()
    pat_index = patients_df.idx.apply(lambda x: x[0]).unique()
    #patients_arr = patients_arr[patients_df.idx.apply(lambda x: x[0]).unique()]
    patients_arr = patients_array.copy()
    patients_arr = patients_arr[pat_index]
    

    print(f"\n{'Splitting'.center(50, '.')}")
    train_ratio,validation_ratio,test_ratio = ratio
    # splitting the data into training, test and validation sets
    #x_train, x_test, train_df, test_df = train_test_split(patients_arr , patients_df.idx.apply(lambda x: x[0]).unique(), train_size=(1-test_ratio), random_state=42, shuffle=True)


    label_split = patients_df.drop_duplicates(["Subject ID", "Study Date"]).ClinSig.dropna().replace({"non-significant": 0, "significant": 1})


    x_train, x_test, train_df, test_df = train_test_split(patients_arr , pat_index, train_size=(1-test_ratio), random_state=42, shuffle=True, stratify = label_split)

    label_split = patients_df[~patients_df.idx.apply(lambda x: x[0] in test_df)].drop_duplicates(["Subject ID", "Study Date"]).ClinSig.dropna().replace({"non-significant": 0, "significant": 1})
    
    x_train, x_val, train_df, val_df  = train_test_split(x_train , train_df, train_size=train_ratio/(train_ratio+validation_ratio),random_state=42, shuffle=True, stratify = label_split)

    # Update the dataframe with the new indexes
    df_idx = np.concatenate([train_df, test_df, val_df])
    split_idx = np.concatenate([np.arange(len(i)) for i in [train_df,test_df,val_df]])
    split_names = np.concatenate([["y_train"]*len(train_df), ["y_test"]*len(test_df), ["y_val"]*len(val_df)])
    
    patients_df[["split","pat_idx"]] = patients_df.idx.apply(lambda x: pd.Series([split_names[df_idx == x[0]][0],(split_idx[df_idx == x[0]][0],x[1])]))

    n_removed = len(pat_index)-len(old_pat_index)#len(patients_df.idx.apply(lambda x: x[0]))
    if n_removed:
        print(f"\nRemoved {n_removed} patients without labels")
        
    print(f"\n|\tTrain\t|\tVal\t|\tTest\t|")
    print(f"|\t{(len(x_train)/len(patients_arr))*100:.0f}%\t|\t{(len(x_val)/len(patients_arr))*100:.0f}%\t|\t{(len(x_test)/len(patients_arr))*100:.0f}%\t|")
    print(f"|\t{len(x_train)}\t|\t{len(x_val)}\t|\t{len(x_test)}\t|")

    subjectid = ["Subject ID", "Study Date"]
    print(f'\nTotal - {[f"{k}:{v}" for k,v in patients_df.drop_duplicates(subjectid).ClinSig.value_counts().items()]}')
    
    [print(f'{df} - {[f"{k}:{v} - ({(v/patients_df.drop_duplicates(subjectid).ClinSig.value_counts()[k])*100:.0f}%)" for k,v in patients_df[patients_df.split == df].drop_duplicates(subjectid).ClinSig.value_counts().items()]}') for df in ["y_train","y_val","y_test"]][0]
    
    return x_train, x_test, x_val, patients_df

def train_val(patients_arr, patients_df, ratio):
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
    train_ratio,test_ratio = ratio
    # splitting the data into training, test and validation sets
    x_train, x_val, train_df, val_df = train_test_split(patients_arr , patients_df.idx.apply(lambda x: x[0]).unique(), train_size=train_ratio, random_state=42, shuffle=True)

    # Update the dataframe with the new indexes
    df_idx = np.concatenate([train_df, val_df])
    split_idx = np.concatenate([np.arange(len(i)) for i in [train_df,val_df]])
    split_names = np.concatenate([["y_train"]*len(train_df), ["y_val"]*len(val_df)])
    patients_df[["split","pat_idx"]] = patients_df.idx.apply(lambda x: pd.Series([split_names[df_idx == x[0]][0],(split_idx[df_idx == x[0]][0],x[1])]))
    

    print(f"\n|\tTrain\t|\tVal\t|")
    print(f"|\t{(len(x_train)/len(patients_arr))*100:.0f}%\t|\t{(len(x_val)/len(patients_arr))*100:.0f}%\t|")
    print(f"|\t{len(x_train)}\t|\t{len(x_val)}\t|")


    return x_train, x_val, patients_df


def image_to_np_reshape(train_test_val_split,patients_df,channels=1):
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
            
            reshaped_arr[i] = np.zeros(shape=((patients_arr.shape[0],dim[2],dim[1],dim[0],channels)),dtype=np.float32)

            # reshaped_arr[i] = np.zeros(shape=((patients_arr.shape[0],dim[2],dim[1],dim[0])),dtype=np.float32)

            #reshaped_arr[i] = np.zeros(shape=((patients_arr.shape[0],dim[1],dim[0],dim[2],channels)),dtype=np.float32)
            
        for j,pat in enumerate(patients_arr):
            for k,pat_slices in enumerate(pat):
                
                reshaped_arr[k][j] = tf.repeat(tf.expand_dims(tf.cast(sitk.GetArrayFromImage(pat_slices),tf.float32),-1), channels, -1)
                # reshaped_arr[k][j] = tf.cast(sitk.GetArrayFromImage(pat_slices),tf.float32)
                

                #reshaped_arr[k][j] = tf.repeat(tf.expand_dims(tf.convert_to_tensor(sitk.GetArrayFromImage(pat_slices).transpose(1,2,0)),-1), channels, -1)
        
        for i in range(len(reshaped_arr)):
            # reshaped_arr[i]=tf.convert_to_tensor(tf.expand_dims(reshaped_arr[i],-1),dtype=tf.float32)
            reshaped_arr[i]=tf.convert_to_tensor(reshaped_arr[i],dtype=tf.float32)
        output.append(reshaped_arr)
    
    # Updating the dataframe with new indexes
    patients_df[["tag_idx","pat_idx"]] = patients_df.pat_idx.apply(lambda x: pd.Series([x[1],x[0]]))
    #patients_df.drop(columns="idx", inplace=True)

    print(f"\nConversion and reshape finished {(time.time() - start_time):.0f} s")

    return [*output, patients_df]


def preprocess(parameters, nslices = False):
    """
    Loads the slices from the data path, resamples the slices to the desired resolution, normalizes the
    slices and returns the resampled slices and the dataframe

    If size is set to None, then the most common size will be used
        Examples:
            tags = {"ADC":(20,86,128),"t2tsetra": (20,320,320)} 

            tags = {"T2TSETRA": (32,320,320), "adc": None} 

            tags = {"ADC": None}     
    
    :param data_path: path to the folder containing the data
    :param tags: a dictionary of tags and their corresponding sizes
    :return: the preprocessed slices and the dataframe with the metadata.
    """

    print(f"Preprocess started".center(50, '_'))
    start_time = time.time()

    pat_slices, pat_df = load_slices(parameters.data_path, parameters.tags, nslices=nslices)
    pat_slices, pat_df = resample_pat(pat_slices, pat_df, parameters.tags)

    pat_slices = normalize(pat_slices, pat_df)

    print("\n"+f"Preprocess finished {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

    return pat_slices, pat_df




def update_shape(pat_df):
    if modality.modality_name.startswith("Merged"):
        shape,idx=[],[]
        for modality_name in modality.merged_modalities:
            shape_,idx_ = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality_name, case=False)].values[0]
            shape.append(shape_)
            idx.append(idx_)
        modality.same_shape = len(set(shape))==1.
    else:
        shape,idx = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality.modality_name, case=False)].values[0]
    
    modality.image_shape = shape
    modality.idx = idx
    
 

def split_data(pat_slices, pat_df, autoencoder = True):
    set_seed()

    if autoencoder:
        y_train, y_val, pat_df = train_val(pat_slices, pat_df, ratio=[0.7,0.3])
        y_train, y_val, pat_df  = image_to_np_reshape([y_train, y_val],pat_df,channels=1)
    else:
        y_train, y_val, y_test, pat_df  = train_test_validation(pat_slices, pat_df, ratio=[0.6,0.1,0.3])
        y_train, y_val, y_test, pat_df  = image_to_np_reshape([y_train, y_test, y_val],pat_df,channels=1)
    
    update_shape(pat_df)

    
    # # shape,idx = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality_name, case=False)].values[0]
    # # #parameters.insert_param(modality_name,"image_shape",shape)
    # # #parameters.insert_param(modality_name,"idx",idx)
    # # modality.modality_name.image_shape = shape
    # # modality.modality_name.idx = idx
    # if modality.modality_name.startswith("Merged"):
    #     shape,idx=[],[]
    #     for modality_name in modality.merged_modalities:
    #         shape_,idx_ = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality_name, case=False)].values[0]
    #         shape.append(shape_)
    #         idx.append(idx_)
    #     modality.same_shape = len(set(shape))==1.
    # else:
    #     shape,idx = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality.modality_name, case=False)].values[0]
    
    # modality.image_shape = shape
    # modality.idx = idx

            
    if autoencoder:
        return y_train, y_val, pat_df
    else: return y_train, y_val, y_test, pat_df
    

    # #shape_idx = {}
    # for modality_name in parameters.tags.keys():
    #     shape,idx = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality_name, case=False)].values[0]
    #     parameters.insert_param(modality_name,"image_shape",shape)
    #     parameters.insert_param(modality_name,"idx",idx)
    #     #shape_idx[modality_name] = {"image_shape":shape, "idx":idx}
    
    # for modality_name in parameters.lst.keys():
    #     if modality_name.startswith("Merged"):
            
    #         # parameters.insert_param(modality_name,"image_shape",[parameters.lst[name]["image_shape"] for name in parameters.lst if not name.startswith("Merged")])
    #         # parameters.insert_param(modality_name,"image_shape",[parameters.lst[name]["image_shape"] for name in parameters.lst[modality_name]["merged_modalities"]])
    #         # parameters.insert_param(modality_name,"idx",[parameters.lst[name]["idx"] for name in parameters.lst[modality_name]["merged_modalities"]])# if not name.startswith("Merged")])
    #         # parameters.insert_param(modality_name,"image_shape",[shape_idx[name]["image_shape"] for name in parameters.lst[modality_name]["merged_modalities"]])
    #         # parameters.insert_param(modality_name,"idx",[shape_idx[name]["idx"] for name in parameters.lst[modality_name]["merged_modalities"]])
            
    #         #print([tuple([shape_idx[name][ishape] for name in parameters.lst[modality_name]["merged_modalities"]]) for ishape in ["image_shape","tag_idx"]])
    #         shape = tuple([parameters.lst[name]["image_shape"] for name in parameters.lst[modality_name]["merged_modalities"]])
    #         parameters.insert_param(modality_name,"image_shape",shape)
    #         parameters.insert_param(modality_name,"same_shape",len(set(shape))==1.)
    #         idx = tuple([parameters.lst[name]["idx"] for name in parameters.lst[modality_name]["merged_modalities"]])
    #         parameters.insert_param(modality_name,"idx",list(idx))

        
            
    # if autoencoder:
    #     return y_train, y_val, pat_df
    # else: return y_train, y_val, y_test, pat_df

    # [parameters.insert_param(modality_name,"image_shape",pat_df["dim"][pat_df.tag.str.contains(modality_name, case=False)].values[0])for modality_name in parameters.lst if not modality_name.startswith("Merged")]
    # [parameters.insert_param(modality_name,"idx",pat_df["tag_idx"][pat_df.tag.str.contains(modality_name, case=False)].values[0]) for modality_name in parameters.lst if not modality_name.startswith("Merged")]

    # [[parameters.insert_param(modality_name,ishape,pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality_name, case=False)].values[0]) for ishape in ["image_shape","idx"] ]for modality_name in parameters.lst if not modality_name.startswith("Merged")]


    # [[parameters.insert_param(modality_name,ishape,pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality_name, case=False)].values[0][i]) for i,ishape in enumerate(["image_shape","idx"]) ]for modality_name in parameters.lst if not modality_name.startswith("Merged")]
    


    # y_train, y_val, y_test  = image_to_np_reshape([y_train, y_val, y_test],pat_df,channels=3)

    # print(y_train.shape)
    # print(y_train[0].shape)
    # print(y_train[0][0].shape)
    # print(y_train[0][0][0].shape)

    # print(type(y_train))
    # print(type(y_train[0]))
    # print(type(y_train[0][0]))
    # print(type(y_train[0][0][0]))
    # print(type(y_train[0][0][0][0]))
    # print(type(y_train[0][0][0][0][0]))

    # print((y_train).shape)
    # print((y_train[0]).shape)
    # print((y_train[0][0]).shape)
    # print((y_train[0][0][0]).shape)
    # print((y_train[0][0][0][0]).shape)
    # print((y_train[0][0][0][0][0]).shape)


#x_train, x_val, x_val = expand_dims([x_train, x_val, x_val],dim=1)
        
    # print("\n"+f"Preprocess finished {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

    # return y_train, y_val, pat_df
