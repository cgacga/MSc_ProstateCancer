# %%

import matplotlib.pyplot as plt
import SimpleITK as sitk

import os
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

np.random.seed(42)

# %%

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
        - Tag
        - idx
        - jdx
    '''
    
    df = pd.read_csv(os.path.normpath(data_path+"metadata.csv"))  

    df = df[df["Series Description"].str.contains("|".join(type_lst))][["Series UID","Subject ID","Series Description","Study Date","File Location"]]

    df["Tag"] = df["Series Description"].str.extract(f"({'|'.join(type_lst)})", expand=False).fillna(df["Series Description"])

    df = df.assign(helpkey=
            df["File Location"].apply(lambda x: int(os.path.splitext(os.path.basename(x))[0])))\
            .sort_values('helpkey').drop('helpkey',axis = 1)

    df = df.groupby(["Subject ID","Study Date","Tag"]).first().sort_values(["Subject ID","Study Date","Tag"],ascending=[True,False,True])

    df["File Location"] = df["File Location"].apply(lambda x: os.path.normpath(data_path+x)) #abspath
    df["idx"] = df.groupby(["Subject ID","Study Date"]).ngroup()
    df["jdx"] = df.groupby(["Tag"]).ngroup()

    print(df.groupby(["Tag"]).size().to_string())

    return df.reset_index()


def load_slices(patients_df):
    patients_arr = []

    for _, group in patients_df.groupby(["Subject ID","Study Date"]):

        pat_slices = []

        for _, patient in group.iterrows():

            reader = sitk.ImageSeriesReader()

            reader.SetFileNames(reader.GetGDCMSeriesFileNames(directory = patient["File Location"], seriesID = patient["Series UID"])[::-1]) #Includes UID in case of multiple dicom sets in the folder & sorting in ascending order

            pat_slices.append(reader.Execute())
        
        patients_arr.append(pat_slices)

    return patients_arr


def normalize(patients_arr):

    minmax_filter = sitk.MinimumMaximumImageFilter()
    adc_max = 0

    t2_upper_perc = []
    t2_lower_perc = []

    for i, pat in enumerate(patients_arr):

        for j, pat_slices in enumerate(pat):

            if j == 0: #to process other types, check the patients_df["jdx"] to obtain the correct index
                t2_array = sitk.GetArrayViewFromImage(pat_slices)
                for img in t2_array:
                    t2_upper_perc.append(np.percentile(img, 99))
                    t2_lower_perc.append(np.percentile(img, 1))
            else :
                minmax_filter.Execute(pat_slices)
                adc_max_intensity = minmax_filter.GetMaximum()
                if adc_max_intensity > adc_max:  
                    adc_max = adc_max_intensity


    cast_image_filter = sitk.CastImageFilter()
    cast_image_filter.SetOutputPixelType(sitk.sitkFloat32)
    normalization_filter = sitk.IntensityWindowingImageFilter()        
    normalization_filter.SetOutputMaximum(1.0)
    normalization_filter.SetOutputMinimum(0.0)

    for i, pat in enumerate(patients_arr):

        for j, pat_slices in enumerate(pat):

            if j == 0:
                normalization_filter.SetWindowMaximum(adc_max)
                normalization_filter.SetWindowMinimum(0.0)
            else :
                normalization_filter.SetWindowMaximum(np.percentile(t2_upper_perc, 99))
                normalization_filter.SetWindowMinimum(np.percentile(t2_lower_perc, 1))

            float_series = cast_image_filter.Execute(pat_slices)
            patients_arr[i][j] = normalization_filter.Execute(float_series)

    return patients_arr


def resample_pat(patients_arr, new_width, new_height, interpolator=sitk.sitkLinear, default_value=0):
    new_spacing = np.zeros(3)

    for i, pat in enumerate(patients_arr):

        for j, pat_slices in enumerate(pat):
            old_size = pat_slices.GetSize()
            old_spacing = pat_slices.GetSpacing()
            new_spacing[0] = old_spacing[0] / new_width * old_size[0]
            new_spacing[1] = old_spacing[1] / new_height * old_size[1]
            # We assume number of slices will be the same one
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
            patients_arr[i][j] = filter.Execute(pat_slices)

    return patients_arr


def train_test_validation(patients_arr, patients_df, train_ratio,test_ratio,validation_ratio):

    # splitting the data into training and test sets
    train_x, test_x, train_df, test_df = train_test_split(patients_arr , patients_df.idx.unique(), train_size=(1-test_ratio), random_state=101, shuffle=True)

    # splitting training data into training and validation sets
    train_x, val_x, train_df, val_df  = train_test_split(train_x , train_df, train_size=train_ratio/(train_ratio+validation_ratio))


    df_idx = np.concatenate([train_df, test_df, val_df])
    split_idx = np.concatenate([np.arange(len(i)) for i in [train_df,test_df,val_df]])
    split_names = np.concatenate([["train_df"]*len(train_df), ["test_df"]*len(test_df), ["val_df"]*len(val_df)])

    patients_df[["Split","idx"]] = patients_df.idx.apply(lambda x: pd.Series([split_names[df_idx == x][0],split_idx[df_idx == x][0]]))

    
    print(f"Train / Test / Validation split : {(len(train_x)/len(patients_arr))*100:.0f} / {(len(test_x)/len(patients_arr))*100:.0f} / {(len(val_x)/len(patients_arr))*100:.0f}")

    return train_x, test_x, val_x, patients_df



def preprocess(path):

    pat_df = parse_csv(path,["t2tsetra","ADC"])

    #print(pat_df.index.tolist())

    #pat_df = pat_df[:10]


    #return load_slices(patient_df[0:50])
    pat_slices = load_slices(pat_df)

    start = time.time()
    pat_slices = normalize(pat_slices)#[0:10])
    end = time.time()
    print(f"time normalize {str(end - start)} s")

    #dimension_check(pat_df)

    #pat_slices = resample_pat(pat_slices,384, 384)

    #train_x, test_x, val_x, pat_df  = train_test_validation(pat_slices, pat_df, 0.7,0.2,0.1)

    return pat_df, pat_slices# train_x, test_x, val_x


data_path = "../Data/manifest-A3Y4AE4o5818678569166032044/"
start = time.time()
# pat_df, train_x, test_x, val_x = preprocess(data_path)
pat_df, pat_slices = preprocess(data_path)
end = time.time()
print(f"time preprocess {str(end - start)} s")

pat_df


# %%

def getsize(patients_arr,df):
    patients_df = df.copy()

    dim_arr = []
    for i, pat in enumerate(patients_arr):
        for j, pat_slices in enumerate(pat):
            dim_arr.append(pat_slices.GetSize()) #https://stackoverflow.com/a/56746204/13071738

    # dim_arr = np.array(dim_arr) 
    # patients_df = patients_df.join(pd.DataFrame({"dim":zip(dim_arr[:,0],dim_arr[:,1]),"z":dim_arr[:,2]}))
    patients_df["dim"] = dim_arr

    wanted_dim = patients_df.groupby(["Tag"],as_index=False)["dim"].value_counts().groupby("Tag").first().reset_index().rename(columns={"dim":"wanted_dim"})#.drop(columns=["count"])#.reset_index()

    patients_df = patients_df.merge(wanted_dim.drop(columns=["count"]), on="Tag", how="left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

    ## wrong_dim = patients_df.merge(wanted_dim, how="outer", indicator=True).query('_merge=="left_only"').drop(columns=["_merge"])
    
    print(patients_df.groupby(["Tag"])["dim"].value_counts())
    #print(patients_df.groupby(["Tag"])["xy","z"].value_counts())
    print(wanted_dim)
    #print(patients_df[patients_df["dim"].ne(patients_df["wanted_dim"])].groupby("Tag")["dim"].value_counts())   

    return patients_df


def resample_pat(patients_arr,patients_df, width=False, height=False, interpolator=sitk.sitkLinear, default_value=0):
    new_spacing = np.zeros(3)

    for _, pat in patients_df.iterrows():
        old_size = patients_arr[pat.idx][pat.jdx].GetSize()
        
        
        if not (width and height):
            if  old_size == pat.wanted_dim: continue
            dim = pat.wanted_dim
            new_width = dim[0]
            new_height = dim[1]
            new_depth = dim[2]

        pat_slices = patients_arr[pat.idx][pat.jdx]
        
        old_spacing = pat_slices.GetSpacing()
        new_spacing[0] = old_spacing[0] / new_width * old_size[0]
        new_spacing[1] = old_spacing[1] / new_height * old_size[1]
        # We assume number of slices will be the same one
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
        patients_arr[pat.idx][pat.jdx] = filter.Execute(pat_slices)

    return  patients_arr

pat_df_asdasd = getsize(pat_slices,pat_df)
qwe_slices = resample_pat(pat_slices,pat_df_asdasd)
asd_df = getsize(qwe_slices,pat_df_asdasd)


# %%

