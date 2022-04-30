
### Data Augmentation ###

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers



def keras_augment(images,ksizes,depth,channels):

    ksize_height, ksize_width = ksizes

    # images = tf.reshape(images, [-1, depth,ksize_height,ksize_width,channels])

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
    if len(patients.shape) < 5:
        patients = tf.expand_dims(patients, axis=0)
    patients_shape = tf.shape(patients)
    depth = patients_shape[1]
    channels = patients_shape[-1]
    depth_range = tf.range(patients_shape[1])
    channels_range = tf.range(patients_shape[-1])
    # reconstructed_arr = np.zeros_like(patients)
    #reconstructed_arr = tf.zeros_like(patients)
    reconstructed_arr = tf.TensorArray(patients.dtype, size=0, dynamic_size=True, clear_after_read=True)
    reconstructed_arr = reconstructed_arr.unstack(patients)
    slice = tf.zeros_like(patients[0])
    
    no_adjacent = True
    trials = 100

    min_reduction = 5
    max_reduction = 15

    min_percentage = 10
    max_percentage = 15
    
    mask_percentage = 50

    for pat,patient in enumerate(patients):
        
        #rnd_reduction = np.random.randint(min_reduction,max_reduction,size=2)
        rnd_reduction = tf.random.uniform(shape=(2,), minval=min_reduction, maxval=max_reduction, dtype=tf.int32)

        patch_height = patients_shape[-3]//rnd_reduction[0]
        patch_width = patients_shape[-2]//rnd_reduction[1]

        
        ksizes = tf.stack([patch_height,patch_width])
        strides = tf.stack([patch_height,patch_width])

        
        patches = Patches(ksizes,strides)(tf.expand_dims(patient, axis=0))

        #num_patches = patches.shape[0]       
        num_patches = tf.shape(patches)[0]
        
        #rnd_percentage = np.random.randint(min_percentage,max_percentage,size=1)
        rnd_percentage = tf.random.uniform(shape=(), minval=min_percentage, maxval=max_percentage, dtype=tf.int32)

        n_augmentations = num_patches*rnd_percentage//100
        
        num_patches_x,num_patches_y = tf.unstack([(patient.shape[i+1] // k)-1  for i,k in enumerate(ksizes)])

        left_border = tf.stack([(num_patches_y+1) * i for i in range(num_patches_x+1)])
        right_border = tf.stack([((num_patches_y+1) * i) - 1 for i in range(1,num_patches_x+2)])
        
        if no_adjacent:
            for mmm in range(trials):
                #idx_augmentations = np.random.choice(num_patches, size=n_augmentations, replace=False)
                idx_augmentations = tf.math.top_k(tf.random.uniform(shape=[num_patches]), n_augmentations, sorted=False).indices
                n_mask = tf.size(idx_augmentations)*mask_percentage//100
                idx_mask = idx_augmentations[:n_mask]
                idx_rotate = idx_augmentations[n_mask:]
                
                valid = True
                for i in [idx_mask,idx_rotate]:
                    
                    elements = tf.TensorArray(i.dtype, size=0, dynamic_size=True, clear_after_read=True)
                    
                    k = 0
                    for j in i:
                        if j not in left_border:
                            elements = elements.write(k,j-1)
                            k = k+ 1
                        if j not in right_border:
                            elements = elements.write(k,j+1)
                            k = k+ 1
                        elements = elements.write(k,j-1-num_patches_y)
                        k = k+ 1
                        elements = elements.write(k,j+1+num_patches_y)
                        k = k+ 1
                       
                    elements_stack = elements.stack()
                    elements = elements.close()
                    #elements = elements.mark_used()

                    elements_stack = tf.boolean_mask(elements_stack,tf.math.greater_equal(elements_stack,0)&tf.math.less(elements_stack,num_patches))
                    
                    if tf.reduce_any(tf.reduce_any(tf.equal(tf.expand_dims(elements_stack, 0), tf.expand_dims(i, 1)), 1)):
                        valid = False
                        break

                    # elements = np.array([])
                    # for j in i:
                    #     if j not in left_border:
                    #         elements = np.append(elements,j-1)
                    #     if j not in right_border:
                    #         elements = np.append(elements,j+1)
                    #     elements = np.append(elements,j-1-num_patches_y)
                    #     elements = np.append(elements,j+1+num_patches_y)
                        
                    # elements = elements[tuple([tf.math.greater_equal(elements,0)&tf.math.less(elements,num_patches)])]
                    
                    # if np.any(np.in1d(i, elements)):
                    #     valid = False
                    #     break
                                    
                if valid:
                    break
            #print("mmm ",mmm)      
            
        patches = tf.tensor_scatter_nd_update(patches, tf.expand_dims(idx_rotate, 1), keras_augment(tf.gather(patches, idx_rotate),ksizes,depth,channels))
        patches = tf.tensor_scatter_nd_update(patches, tf.expand_dims(idx_mask, 1), tf.zeros_like(tf.gather(patches, idx_mask)))

        step_x, step_y = tf.unstack(ksizes)
        x,y = 0,-1
        
        for patch in patches:
            if y >= num_patches_y:
                y = 0
                if x >= num_patches_x:
                    x = 0
                else: x = x+1
            else: 
                y = y+1
        
            x_pos, y_pos = x * step_x, y * step_y

            # i1, i2,i3, i4, i5 = tf.meshgrid(pat, tf.range(patients_shape[1]),tf.range(x_pos,x_pos + step_x),tf.range(y_pos,y_pos + step_y),tf.range(patients_shape[-1]),indexing='ij')
            # idx = tf.stack([i1, i2, i3, i4, i5], axis=-1)
            # reconstructed_arr = tf.tensor_scatter_nd_update(reconstructed_arr, idx, tf.expand_dims(patch, 0))

            i2,i3, i4, i5 = tf.meshgrid(depth_range ,tf.range(x_pos,x_pos + step_x),tf.range(y_pos,y_pos + step_y),channels_range,indexing='ij')
            idx = tf.stack([i2, i3, i4, i5], axis=-1)

            slice = tf.tensor_scatter_nd_update(slice, idx, patch)
        reconstructed_arr = reconstructed_arr.write(pat,slice)
    reconstructed_arr = reconstructed_arr.stack()
            
            # reconstructed_arr[pat,:,x_pos:x_pos + step_x, y_pos:y_pos + step_y,:] = patch

    return reconstructed_arr


def augment_build_datasets(y_train,y_val, batch_size):

    print(f"Augment and build dataset started".center(50, '_'))
    start_time = time.time()

    train_loader = tf.data.Dataset.from_tensor_slices((augment_patches(y_train), y_train))
    val_loader = tf.data.Dataset.from_tensor_slices((augment_patches(y_val), y_val))
    
    #batch_size = 32

    #32 -> 34147MiB / 40536MiB (2 gpu -> 16 each)
    #16 ->  34147MiB / 40536MiB 
    #8  -> 17819MiB / 40536MiB
    print(f"Batch size = {batch_size}")


    trainDS = (
        train_loader
            .batch(
                batch_size = batch_size
                ,num_parallel_calls=tf.data.AUTOTUNE)
            .map(
                lambda x, y: (tf.repeat(x,3,-1), y)#tf.repeat(y,3,-1))
                ,num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(
                buffer_size = tf.data.AUTOTUNE))
    valDS = (
        val_loader
            .batch(
                batch_size = batch_size
                ,num_parallel_calls=tf.data.AUTOTUNE)
            .map(
                lambda x, y: (tf.repeat(x,3,-1), y)#tf.repeat(y,3,-1))
                ,num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(
                buffer_size = tf.data.AUTOTUNE))

    gpus = tf.config.list_physical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy()
    if len(gpus)>1:
        trainDS = strategy.experimental_distribute_dataset(train_loader)
        valDS = strategy.experimental_distribute_dataset(val_loader)


    print("\n"+f"Augment and build finished {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

    return trainDS, valDS




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

