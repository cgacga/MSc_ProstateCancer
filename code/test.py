#%%
import os, sys, time, random
import sys, importlib
import numpy as np
import tensorflow as tf

# from tensorflow.keras import backend as K

from preprocess import *
from data_augmentation import *
from model_building import *
from img_display import *

def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(s) 
seed_all(42)


data_path = "../data/manifest-A3Y4AE4o5818678569166032044/"
#tags = {"ADC": None,"t2tsetra": None} 
#tags = {"ADC": None,"t2tsetra": (32,320,320)} 
#tags = {"t2tsetra": (32,320,320)} 
tags = {"t2tsetra": None} 


#%%

import importlib
importlib.reload(sys.modules['preprocess'])
importlib.reload(sys.modules['model_building'])
importlib.reload(sys.modules['img_display'])
from preprocess import *
from model_building import *
from img_display import *



import timeit
n = 10
result = timeit.timeit(stmt='preprocess(data_path,tags,True)', globals=globals(), number=n)
print(f"Execution time is {result / n} seconds")

# y_train, y_test, pat_df = preprocess(data_path,tags,True)

# x_train, x_test, x_val, y_train, y_test, y_val = data_augmentation(pat_slices, pat_df)

# for modality in tags.keys():
#     shape,idx = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality, case=False)].values[0]
#     img_pltsave([y_train[idx][0], y_test[idx][0]])

#%%

y_train[0].dtype
y_train[0].shape

#img_pltsave([tf.sparse.to_dense(y_train[0][0])])

type(y_train[0])
#%%
w = tf.sparse.to_dense(y_train[0])
q = tf.sparse.to_dense(y_test[0])
for modality in tags.keys():
    shape,idx = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality, case=False)].values[0]
    img_pltsave([w[0],q[0]])

#%%


def keras_augment(images,ksizes,depth=32,channels=3):

    ksize_height, ksize_width = ksizes

    images = tf.reshape(images, [-1, depth,ksize_height,ksize_width,channels])
        
    data_augmentation = tf.keras.Sequential(
        [
            layers.Permute(dims=(2,3,1,4)),
            layers.Reshape((ksize_height,ksize_width,depth*channels)),

            #layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.083334,fill_mode="constant",fill_value=0),
            #2 * pi * (0.1 rad) = 36 deg
            #2 * pi * (0.08333(repeating ofc) rad) /approx 30 deg
            
            layers.Reshape(target_shape=(ksize_height,ksize_width,depth,channels)),
            layers.Permute(dims=(3,1,2,4)),
        ],
        name="data_augmentation",
    )
    images = data_augmentation(images)
    
    return images



class Patches(layers.Layer):
    def __init__(self, ksizes = [32,32], strides = [32,32]):
        super(Patches, self).__init__()
        self.ksize_height, self.ksize_width = ksizes
        self.strides_height, self.strides_width =strides

    def call(self, images):
                
        planes = tf.shape(images)[1]
        channels = tf.shape(images)[-1]

        patches = tf.extract_volume_patches(
            input=images,
            ksizes=[1, planes, self.ksize_height, self.ksize_width, 1],
            strides=[1, planes, self.strides_height, self.strides_width, 1],
            padding="VALID",
        )
        
        patches = tf.reshape(patches, [-1, planes,self.ksize_height,self.ksize_width,channels])
                
        return patches


def augment_patches(patients):

    
    img_heigth = patients.shape[-3]
    img_width = patients.shape[-2]
    reconstructed_arr = np.zeros_like(patients)

    no_adjacent = True
    trials = 100

    for pat,patient in enumerate(patients):

        #må være noe med min og max størrelse på patch i forhold til max patch
        
        min_reduction = 5
        max_reduction = 15

        min_percentage = 10
        max_percentage = 15
        
        mask_percentage = 50

        
        rnd_reduction = np.random.randint(min_reduction,max_reduction,size=2)
        patch_height = int(img_heigth/rnd_reduction[0])
        patch_width = int(img_width/rnd_reduction[1])

        ksizes = [patch_height,patch_width]
        strides = [patch_height,patch_width]
        
        patches = Patches(ksizes,strides)(tf.expand_dims(patient, axis=0))
        num_patches = patches.shape[0]       
        rnd_percentage = np.random.randint(min_percentage,max_percentage,size=1)
        n_augmentations = num_patches*rnd_percentage//100
        idx_augmentations = np.random.choice(num_patches, size=n_augmentations, replace=False)
        num_patches_x,num_patches_y = [(patient.shape[i+1] // k)-1  for i,k in enumerate(ksizes)]
        left_border = [(num_patches_y+1) * i for i in range(num_patches_x+1)]
        right_border = [((num_patches_y+1) * i) - 1 for i in range(1,num_patches_x+2)]
        
        if no_adjacent:
            for mmm in range(trials):
                idx_augmentations = np.random.choice(num_patches, size=n_augmentations, replace=False)
                n_mask = len(idx_augmentations)*mask_percentage//100
                idx_mask = idx_augmentations[:n_mask]
                idx_rotate = idx_augmentations[n_mask:]
                
                valid = True
                for i in [idx_mask,idx_rotate]:
                    elements = np.array([])
                    for j in i:
                        if j not in left_border:
                            elements = np.append(elements,j-1)
                        if j not in right_border:
                            elements = np.append(elements,j+1)
                        elements = np.append(elements,j-1-num_patches_y)
                        elements = np.append(elements,j+1+num_patches_y)
                    elements = elements[(elements>=0)&(elements<num_patches)]
                    
                    if np.any(np.in1d(i, elements)):
                        valid = False
                        # print("restart")
                        # print("i ",i)
                        # print(np.in1d(i, elements))
                        break
                                    
                if valid:
                    break
            #print("mmm ",mmm)      
            
        patches = tf.tensor_scatter_nd_update(patches, tf.expand_dims(idx_rotate, 1), keras_augment(tf.gather(patches, idx_rotate),ksizes))
        patches = tf.tensor_scatter_nd_update(patches, tf.expand_dims(idx_mask, 1), tf.zeros_like(tf.gather(patches, idx_mask)))

        step_x, step_y = ksizes
        x,y = 0,-1
        for i,patch in enumerate(patches):
            if y >= num_patches_y:
                y = 0
                if x >= num_patches_x:
                    x = 0
                else: x = x+1
            else: 
                y = y+1
        
            x_pos, y_pos = x * step_x, y * step_y
        
            reconstructed_arr[pat,:,x_pos:x_pos + step_x, y_pos:y_pos + step_y,:] = patch

    return reconstructed_arr

    
    #[img_pltsave([reconstructed_arr[i],asd[i]]) for i in range(reconstructed_arr.shape[0])]
    
    #img_pltsave([reconstructed_arr[0]])
    #img_pltsave([reconstructed_arr[1]])

# import timeit
# n = 10
# result = timeit.timeit(stmt='augment_patches(asd)', globals=globals(), number=n)
# print(f"Execution time is {result / n} seconds")

asd = y_train[0]
reconstructed_arr = augment_patches(asd)
img_pltsave([arr for arr in reconstructed_arr])
#img_pltsave([arr for arr in asd])

#%%
#img_pltsave([arr for arr in asd])
    

img_pltsave([[reconstructed_arr[i],asd[i]] for i in range(reconstructed_arr.shape[0])])
