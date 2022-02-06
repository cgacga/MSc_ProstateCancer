# %%

#### %matplotlib inline
from attr import asdict
import matplotlib.pyplot as plt
import SimpleITK as sitk

import os
import sys


import numpy as np
import pandas as pd

# %%


def parse_csv(data_path,type_lst):
    '''
    This function takes in a path to a metadata.csv file and a list of strings that you want to search for. 
    It returns a dataframe with the subject ID, study date, tag and file location for each file that contains one of the strings in the list.
    
    :param data_path: the path to the data directory
    :param type_lst: list of strings that are the series descriptions you want to include in the dataset
    :return: A dataframe with the following columns:
        - Subject ID
        - Study Date
        - Tag
        - Series Description
        - File Location
    '''
    
    df = pd.read_csv(os.path.normpath(data_path+"metadata.csv"))  

    df = df[df["Series Description"].str.contains("|".join(type_lst))][["Series UID","Subject ID","Series Description","Study Date","File Location"]]

    df["Tag"] = df["Series Description"].str.extract(f"({'|'.join(type_lst)})", expand=False).fillna(df["Series Description"])

    df = df.assign(helpkey=
            df["File Location"].apply(lambda x: int(os.path.splitext(os.path.basename(x))[0])))\
            .sort_values('helpkey').drop('helpkey',axis = 1)

    df = df.groupby(["Subject ID","Study Date","Tag"]).first().sort_values(["Subject ID","Study Date","Tag"],ascending=[True,False,False])


    df["File Location"] = df["File Location"].apply(lambda x: os.path.normpath(data_path+x)) #abspath

    print(df.groupby(["Tag"]).size().to_string())

    return df.reset_index()


data_path = "../Data/manifest-A3Y4AE4o5818678569166032044/"

type_lst = ["t2tsetra","ADC"]

patient_df = parse_csv(data_path,type_lst)


# %%

def load_and_normalize(patient_df):


    for index, patient in patient_df.iterrows():

      

        reader = sitk.ImageSeriesReader()
        


        reader.SetFileNames(reader.GetGDCMSeriesFileNames(directory = patient["File Location"], seriesID = patient["Series UID"])[::-1]) #Includes UID in case of multiple dicom sets in the folder #ascending order

        image_series = reader.Execute()


        img_array = sitk.GetArrayViewFromImage(image_series)

    return image_series

    

asd = load_and_normalize(patient_df[0:1])


print(asd)



# %%

