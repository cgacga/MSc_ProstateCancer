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

    df = df[df["Series Description"].str.contains("|".join(type_lst))][["Subject ID","Series Description","Study Date","File Location"]]

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

#patient_df.to_clipboard()

patient_df

# %%

def extract_dicom_paths(patient_df):



    paths = patient_df["File Location"].tolist()



    for path in paths:

        reader = sitk.ImageSeriesReader
        sorted_file_names = reader.GetGDCMSeriesFileNames(path)




        print(sorted_file_names)

        # Read the bulk pixel data
        for i in sorted_file_names:
            img = sitk.ReadImage(i)


            img_array = sitk.GetArrayViewFromImage(img)


            img_array = img_array[0,:,:]
            plt.imshow(img_array,cmap="gray")
            plt.show()




        # with os.scandir(path) as listOfEntries:
            
            
        #     for entry in sorted(listOfEntries, key=lambda x: x.name):
        #         # print all entries that are files
        #         #if entry.is_file():
        #         if entry.name.endswith('.dcm'):
        #             print(entry.name)
        # print("\n")
  

    #for root,dirs,files in os.walk(paths):
    #    print(files)

    return 2

extract_dicom_paths(patient_df[0:1])

# %%

