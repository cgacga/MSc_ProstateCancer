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
#tags = {"ADC": None,"t2tsetra": (32,320,320)} 
tags = {"t2tsetra": (32,320,320)} 


import importlib
#%%


importlib.reload(sys.modules['preprocess'])
importlib.reload(sys.modules['model_building'])
importlib.reload(sys.modules['img_display'])
from preprocess import *
from model_building import *
from img_display import *

y_train, y_test, pat_df = preprocess(data_path,tags,True)
# x_train, x_test, x_val, y_train, y_test, y_val = data_augmentation(pat_slices, pat_df)


for modality in tags.keys():
    shape,idx = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality, case=False)].values[0]
    img_pltsave([y_train[idx][0], y_test[idx][0]])


#%%


def keras_augment(images,ksizes,depth=32,channels=3):

    patch_dims = tf.shape(images)[-1]
    batch_size = tf.shape(images)[0]
    ksize_rows, ksize_cols = ksizes

    images = tf.reshape(images, [-1, depth,ksize_rows,ksize_cols,channels])
    # images = tf.transpose(images, perm=[0,2,3,1,4])
    # images = tf.reshape(images, [-1, 32,32,32*3])
    
    kwargs={"data_format": "channels_first"}

    # def random_flip_on_probability(image, probability= 0.5):
    #     if random.random() < probability:
    #         # return tf.image.random_flip_left_right(image)
    #         #return layers.RandomRotation(factor=0.02,fill_mode="constant",fill_value=0)
    #         return tf.keras.preprocessing.image.random_rotation(image,rg=55.02,fill_mode="constant",cval=0)
    #     return image

    data_augmentation = tf.keras.Sequential(
        [
            layers.Permute(dims=(2,3,1,4)),
            layers.Reshape((ksize_rows,ksize_cols,depth*channels)),

            #layers.RandomFlip("horizontal_and_vertical"),
            #layers.RandomFlip("horizontal"),
            # layers.Lambda(random_flip_on_probability),
            layers.RandomRotation(factor=0.02,fill_mode="constant",fill_value=0),
            
            layers.Reshape(target_shape=(ksize_rows,ksize_cols,depth,channels)),
            layers.Permute(dims=(3,1,2,4)),
        ],
        name="data_augmentation",
    )

    print(images.shape)

    images = data_augmentation(images)

    print(images.shape)
    
    # images = tf.reshape(images, [-1, 32, 32, 32,3])
    # images = tf.transpose(images, perm=[0,3,1,2,4])
    #images = tf.reshape(images, [batch_size,-1, patch_dims])
    return images


class Patches(layers.Layer):
    def __init__(self, ksizes = [32,32], strides = [32,32]):
        super(Patches, self).__init__()
        self.ksize_rows, self.ksize_cols = ksizes
        self.strides_rows, self.strides_cols =strides

    def call(self, images):
        planes = tf.shape(images)[1]

        patches = tf.extract_volume_patches(
            input=images,
            ksizes=[1, planes, self.ksize_rows, self.ksize_cols, 1],
            strides=[1, planes, self.strides_rows, self.strides_cols, 1],
            padding="SAME",
        )
        # patch_dims = patches.shape[-1]
        #patches = tf.reshape(patches, [batch_size,-1, patch_dims])
        return patches



class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


asd = y_train[0]

print(asd.shape)



image_size = (320)
patch_size = 32  # Size of the patches to be extract from the input images
cube = 32
stride = 32
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
print(num_patches)



ksizes = [32*3,32*3]
strides = [32*3,32*3]
# ksizes = [32,32]
# strides = [32,32]


patches = Patches(ksizes,strides)(asd)


patches = keras_augment(patches,ksizes)


# encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)



import importlib
import img_display
importlib.reload(img_display)
from img_display import *


img_pltsave([asd[0]])


patch_pltsave(patches[0],ksizes)

#patch_pltsave(encoded_patches[0],ksizes)


#%%


from skimage.util.shape import view_as_windows

from skimage.util import view_as_blocks
#%%

print(patches.shape)

#%%

size = 32 # patch size

qwe = view_as_windows(patches.numpy(), (32,size,size))#[...,0,:,:]
print(qwe.shape)

zxc = view_as_blocks(patches.numpy(), (1,size,size))#[...,0,:,:]
print(zxc.shape)

#%%

reconstructed_arr = np.zeros_like(asd)

print("patches ",patches.shape)
print("reconstructed_arr ",reconstructed_arr.shape)
step = ksizes[0]
for x in range(reconstructed_arr.shape[1]):
        for y in range(reconstructed_arr.shape[2]):
            x_pos, y_pos = x * step, y * step
            reconstructed_arr[x_pos:x_pos + 128, y_pos:y_pos + 128] = patches[0,x, y,  ...]



#%%
#num_patches_x,num_patches_y = [(asd.shape[i+2] // k) ** 2 for i,k in enumerate(ksizes)]
num_patches_x,num_patches_y = [(asd.shape[i+2] // k)  for i,k in enumerate(ksizes)]

#for i,k in enumerate(ksizes):
#    print(asd.shape[i+1])
#    print(k)

print("num_patches_x ", num_patches_x)
print("num_patches_y ",num_patches_y)

#%%
reconstructed_arr = np.zeros_like(asd)
num_patches_x,num_patches_y = [(asd.shape[i+2] // k)  for i,k in enumerate(ksizes)]

print("patches ",patches.shape)
print("reconstructed_arr ",reconstructed_arr.shape)


step_x, step_y = ksizes
i = 0

for pat in range(reconstructed_arr.shape[0]):
    #for z in range(reconstructed_arr.shape[1]):
    #for x in range(reconstructed_arr.shape[2]):
    #    for y in range(reconstructed_arr.shape[3]):
    for x in range(num_patches_x):
        for y in range(num_patches_y):
            x_pos, y_pos = x * step_x, y * step_y
            # print("y ",y)
            # print("x ",x)
            # print("y_pos + step_y ",y_pos + step_y)
            
            reconstructed_arr[pat,:,x_pos:x_pos + step_x, y_pos:y_pos + step_y] = patches[i]
            i = i+1


reconstructed_arr.shape


img_pltsave([reconstructed_arr[0]])

#%%

num_patches_x,num_patches_y = [(asd.shape[i+2] // k)  for i,k in enumerate(ksizes)]

print("num_patches_x ", num_patches_x)
print("num_patches_y ",num_patches_y)


#%%


reconstructed_arr = np.zeros_like(asd)
num_patches_x,num_patches_y = [(asd.shape[i+2] // k)  for i,k in enumerate(ksizes)]

print("patches ",patches.shape)
print("reconstructed_arr ",reconstructed_arr.shape)


step_x, step_y = ksizes
# i = 0

# for pat in range(reconstructed_arr.shape[0]):
#     #for z in range(reconstructed_arr.shape[1]):
#     #for x in range(reconstructed_arr.shape[2]):
#     #    for y in range(reconstructed_arr.shape[3]):
#     for x in range(num_patches_x):
#         for y in range(num_patches_y):
#             x_pos, y_pos = x * step_x, y * step_y
#             # print("y ",y)
#             # print("x ",x)
#             # print("y_pos + step_y ",y_pos + step_y)
            
#             reconstructed_arr[pat,:,x_pos:x_pos + step_x, y_pos:y_pos + step_y] = patches[i]
#             i = i+1

pat,x,y = 0,0,-1
#for patch,x,y in zip(patches,range(num_patches_x),range(num_patches_y)):
#for patch,y in zip(patches,range(num_patches_y)):
for patch in patches:
    
    if x >= num_patches_x and y >= num_patches_y:
        pat = pat+1

    if y >= num_patches_y:
        y = 0
        x = x+1
    else: y = y+1
    
    if x >= num_patches_x:
        x = 0
 
    print("x ",x)
    print("y ",y)
    x_pos, y_pos = x * step_x, y * step_y
    reconstructed_arr[pat,:,x_pos:x_pos + step_x, y_pos:y_pos + step_y] = patch
    #y = y+1




reconstructed_arr.shape


img_pltsave([reconstructed_arr[0]])

#%%

img_pltsave([arr for arr in reconstructed_arr])