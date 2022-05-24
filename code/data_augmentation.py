
### Data Augmentation ###

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import segmentation_models_3D as sm
from params import modality
from model_building import PlotCallback
from main import set_seed


def keras_augment(images,ksizes,depth,channels):

    return keras.Sequential(
        [
            layers.Permute(dims=(2,3,1,4)),
            layers.Reshape((ksizes[0],ksizes[1],depth*channels)),

            #layers.RandomFlip("horizontal_and_vertical"),
            #layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.08333,fill_mode="constant",fill_value=0),
            #2 * pi * (0.1 rad) = 36 deg
            #2 * pi * (0.08333(repeating ofc) rad) /approx 30 deg
            
            layers.Reshape((ksizes[0],ksizes[1],depth,channels)),
            layers.Permute(dims=(3,1,2,4)),
        ],
        name="data_augmentation",
    )(images)



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
    set_seed(42)
    #print("augment")
    #print(patients.shape)
    #if len(patients.shape) < 5:
    #    patients = tf.expand_dims(patients, axis=0)
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
    
    no_adjacent = modality.no_adjacent
    trials = 100 if no_adjacent else 1

    min_reduction = modality.minmax_shape_reduction[0]#5
    max_reduction = modality.minmax_shape_reduction[1]#15

    min_percentage = modality.minmax_augmentation_percentage[0]#10
    max_percentage = modality.minmax_augmentation_percentage[1]#15
    
    mask_percentage = modality.mask_vs_rotation_percentage#50

    for pat,patient in enumerate(patients):
        
        #rnd_reduction = np.random.randint(min_reduction,max_reduction,size=2)
        if min_reduction == max_reduction:
            rnd_reduction = tf.stack([min_reduction,max_reduction])
        else:
            rnd_reduction = tf.random.uniform(shape=(2,), minval=min_reduction, maxval=max_reduction, dtype=tf.int32)

        patch_height = patients_shape[-3]//rnd_reduction[0]
        patch_width = patients_shape[-2]//rnd_reduction[1]

        
        ksizes = tf.stack([patch_height,patch_width])
        strides = tf.stack([patch_height,patch_width])

        
        patches = Patches(ksizes,strides)(tf.expand_dims(patient, axis=0))

        #num_patches = patches.shape[0]       
        num_patches = tf.shape(patches)[0]
        
        #rnd_percentage = np.random.randint(min_percentage,max_percentage,size=1)
        if min_percentage == max_percentage:
            rnd_percentage = min_percentage
        else:
            rnd_percentage = tf.random.uniform(shape=(), minval=min_percentage, maxval=max_percentage, dtype=tf.int32)

        n_augmentations = num_patches*rnd_percentage//100
        
        num_patches_x,num_patches_y = tf.unstack([(patient.shape[i+1] // k)-1  for i,k in enumerate(ksizes)])

        left_border = tf.stack([(num_patches_y+1) * i for i in range(num_patches_x+1)])
        right_border = tf.stack([((num_patches_y+1) * i) - 1 for i in range(1,num_patches_x+2)])
        
        #if no_adjacent:
        if True:
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
    #print(reconstructed_arr.shape)
    # if (reconstructed_arr.shape[-1] == 1):
    #     reconstructed_arr = tf.repeat(reconstructed_arr,3,-1)
    # print(reconstructed_arr.shape)
    
    return reconstructed_arr


def augment_build_datasets(y_train,y_val):

    print(f"Augment and build dataset started".center(50, '_'))
    start_time = time.time()

    #modality.steps_pr_epoch = len(y_train)//modality.batch_size_prgpu
    #modality.validation_steps = len(y_val)//modality.batch_size_prgpu

    #with modality.strategy.scope():
    # train_loader = tf.data.Dataset.from_tensor_slices((augment_patches(y_train), y_train))
    # val_loader = tf.data.Dataset.from_tensor_slices((augment_patches(y_val), y_val))

    #pre = sm.get_preprocessing(modality.backbone_name)
    # y_train = pre(tf.repeat(y_train,3,-1))
    # y_val = pre(tf.repeat(y_val,3,-1))

    # y_train = tf.repeat(y_train,3,-1)
    # y_val = tf.repeat(y_val,3,-1)

    # train_loader = tf.data.Dataset.from_tensor_slices((augment_patches(y_train), y_train))
    # val_loader = tf.data.Dataset.from_tensor_slices((augment_patches(y_val), y_val))
    # yt = pre(tf.repeat(augment_patches(y_train),3,-1))
    # train_loader = tf.data.Dataset.from_tensor_slices((yt, y_train))
    # del yt
    # yv  = pre(tf.repeat(augment_patches(y_val),3,-1))
    # val_loader = tf.data.Dataset.from_tensor_slices((yv, y_val))
    # del yv
    # train_loader = tf.data.Dataset.from_tensor_slices((tf.repeat(augment_patches(y_train),3,-1), y_train))
    # val_loader = tf.data.Dataset.from_tensor_slices((tf.repeat(augment_patches(y_val),3,-1), y_val))


    if isinstance(y_train, np.ndarray):
        
        # train_loader = tf.data.Dataset.from_tensor_slices(({f"input_{i+1}_{modality.merged_modalities[i]}":tf.repeat(augment_patches(y),3,-1) for i,y in enumerate(y_train)},{f"{modality.merged_modalities[i]}":y for i,y in enumerate(y_train)}))

        # val_loader = tf.data.Dataset.from_tensor_slices(({f"input_{i+1}_{modality.merged_modalities[i]}":tf.repeat(augment_patches(y),3,-1) for i,y in enumerate(y_val)},{f"{modality.merged_modalities[i]}":y for i,y in enumerate(y_val)}))


        train_loader = tf.data.Dataset.from_tensor_slices(({f"input_{i+1}_{modality.merged_modalities[i]}":augment_patches(y) for i,y in enumerate(y_train)},{f"{modality.merged_modalities[i]}":y for i,y in enumerate(y_train)}))

        val_loader = tf.data.Dataset.from_tensor_slices(({f"input_{i+1}_{modality.merged_modalities[i]}":augment_patches(y) for i,y in enumerate(y_val)},{f"{modality.merged_modalities[i]}":y for i,y in enumerate(y_val)}))

        # tl_x = {}
        # tl_y = {}
        # for i,y in enumerate(y_train):
        #     tl_x[f"input_{i+1}_{modality.merged_modalities[i]}"] = augment_patches(y)
        #     tl_y[f"{modality.merged_modalities[i]}"] = y

        # train_loader = tf.data.Dataset.from_tensor_slices((tl_x,tl_y))
        # del tl_x,tl_y


        # vl_x = {}
        # vl_y = {}
        # for i,y in enumerate(y_val):
        #     vl_x[f"input_{i+1}_{modality.merged_modalities[i]}"] = augment_patches(y)
        #     vl_y[f"{modality.merged_modalities[i]}"] = y

        # val_loader = tf.data.Dataset.from_tensor_slices((vl_x,vl_y))
        # del vl_x,vl_y

        PlotCallback.x_val = [augment_patches(y[0:modality.tensorboard_num_predictimages]) for y in y_val]

        def repeatfn(x,y):
            return (({key:tf.repeat(val,3,-1) for key,val in x.items()},y))


        trainDS = (
            train_loader
                .batch(
                    batch_size = modality.batch_size
                    ,num_parallel_calls=tf.data.AUTOTUNE)
                .map(
                    #lambda x, y: (tf.repeat(x,3,-1), y)
                    #lambda x,y: ({key:tf.repeat(val,3,-1) for key,val in x.items()},y)
                    #lambda x,y:({key:tf.repeat(val,3,-1) for key,val in x.items()},{key:val for key,val in y.items()})
                    repeatfn
                    ,num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(
                    buffer_size = tf.data.AUTOTUNE)
                    )
        valDS = (
            val_loader
                .batch(
                    batch_size = modality.batch_size
                    ,num_parallel_calls=tf.data.AUTOTUNE)
                .map(
                    #lambda x, y: (tf.repeat(x,3,-1), y)#tf.repeat(y,3,-1))
                    #lambda x,y: (({key:tf.repeat(val,3,-1) for key,val in x.items()},{key:val for key,val in y.items()}))
                    repeatfn
                    ,num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(
                    buffer_size = tf.data.AUTOTUNE)
                    )

    else:

        train_loader = tf.data.Dataset.from_tensor_slices((augment_patches(y_train), y_train))
        val_loader = tf.data.Dataset.from_tensor_slices((augment_patches(y_val), y_val))
        PlotCallback.x_val = augment_patches(y_val[0:modality.tensorboard_num_predictimages])


        
        # pre = sm.get_preprocessing(modality.backbone_name)
        # #tl = pre(tf.repeat(augment_patches(y_train),3,-1))
        # #asd = tf.zeros_like(y_train)
        # train_loader = tf.data.Dataset.from_tensor_slices((pre(tf.repeat(augment_patches(y_train),3,-1)), y_train))
        # # train_loader = tf.data.Dataset.from_tensor_slices((tl, y_train))
        # # del tl
        # val_loader = tf.data.Dataset.from_tensor_slices((pre(tf.repeat(augment_patches(y_val),3,-1)), y_val))
        # PlotCallback.x_val = pre(tf.repeat(augment_patches(y_val[0:modality.tensorboard_num_predictimages]),3,-1))

        #PlotCallback.x_val = list(val_loader.map(lambda x,y: (x[0:modality.tensorboard_num_predictimages])))[0]
        #PlotCallback.x_val = list(val_loader.map(lambda x,y: tf.repeat(x,3,-1)))[0:modality.tensorboard_num_predictimages]
        #PlotCallback.x_val = list(val_loader.map(lambda x,y: x))[0:modality.tensorboard_num_predictimages]
        #PlotCallback.x_val = pre(tf.repeat(augment_patches(y_val[0:modality.tensorboard_num_predictimages]),3,-1))
        
        #PlotCallback.x_val = list(val_loader.map(lambda x,y: pre(tf.repeat(augment_patches(x),3,-1))))[0:modality.tensorboard_num_predictimages]
    
        
        trainDS = (
            train_loader
                .batch(
                    batch_size = modality.batch_size
                    ,num_parallel_calls=tf.data.AUTOTUNE)
                .map(
                    lambda x, y: (tf.repeat(x,3,-1), y)
                    ,num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(
                    buffer_size = tf.data.AUTOTUNE)
                    )
        valDS = (
            val_loader
                .batch(
                    batch_size = modality.batch_size
                    ,num_parallel_calls=tf.data.AUTOTUNE)
                .map(
                    lambda x, y: (tf.repeat(x,3,-1), y)
                    ,num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(
                    buffer_size = tf.data.AUTOTUNE)
                    )

    # gpus = tf.config.list_physical_devices('GPU')
    # strategy = tf.distribute.MirroredStrategy()
    # if len(gpus)>1:
    #PlotCallback.x_val = list(valDS.map(lambda x,y: (x[0:modality.tensorboard_num_predictimages])))[0]#list(valDS.as_numpy_iterator())[0][0][0:modality.tensorboard_num_predictimages]

    # if modality.n_gpus>1:
    #     # Disable AutoShard.
    #     options = tf.data.Options()
    #     options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        
    #     trainDS = trainDS.with_options(options)
    #     valDS = valDS.with_options(options)

    #     trainDS = modality.strategy.experimental_distribute_dataset(trainDS)
    #     valDS = modality.strategy.experimental_distribute_dataset(valDS)



    print("\n"+f"Augment and build finished {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

    return trainDS, valDS


