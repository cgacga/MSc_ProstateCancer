#%%
import os, sys, time, random
import sys, importlib

#from tkinter import E
#from turtle import right
import numpy as np
import tensorflow as tf

# from tensorflow.keras import backend as K

from keras.layers import concatenate, Input

from merged_model import *
from preprocess import *
from data_augmentation import *
from model_building import *
from img_display import *
from params import *



def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(s) 
seed_all(42)


# import importlib
# importlib.reload(sys.modules['preprocess'])
# importlib.reload(sys.modules['model_building'])
# importlib.reload(sys.modules['img_display'])
# importlib.reload(sys.modules['params'])
# importlib.reload(sys.modules['merged_model'])
# from preprocess import *
# from model_building import *
# from img_display import *
# from params import *
# from merged_model import *

# importlib.reload(sys.modules['data_augmentation'])
# from data_augmentation import *

# import gc



# import timeit
# n = 10
# result = timeit.timeit(stmt='pre
# process(data_path,tags,True)', globals=globals(), number=n)
# print(f"Execution time is {result / n} seconds")



task_idx  = 1
epochs = 2
encoder_weights = "imagenet"
encoder_freeze = True
no_adjacent = False

backbone_name = "vgg16"#"vgg16"#, "resnet18"]
#backbone_name = "resnet18"
activation = "sigmoid"#, "softmax"]
decoder_block_type = "upsampling"#"upsampling"#, "transpose"]
learning_rate = 1e-6#1e-2#, 1e-4, 1e-6]


#minmax_augmentation_percentage  = [15,16]
minmax_augmentation_percentage  = [50,50]
minmax_shape_reduction  = [5,5]
mask_vs_rotation_percentage = 100


job_name = f"{backbone_name}_{activation}_{decoder_block_type}_lr{learning_rate}_sr{'-'.join(map(str, minmax_shape_reduction))}_ap{'-'.join(map(str, minmax_augmentation_percentage))}_mvrp{mask_vs_rotation_percentage}_ef{encoder_freeze}_{task_idx}"
#_{encoder_weights}


parameters.set_global(
    data_path="../data/manifest-A3Y4AE4o5818678569166032044/", 
    job_name = job_name,
    backbone_name = backbone_name,
    activation = activation,
    encoder_weights = encoder_weights,
    encoder_freeze = encoder_freeze,
    decoder_block_type = decoder_block_type, 
    epochs = epochs,
    learning_rate  = learning_rate,
    minmax_shape_reduction  = minmax_shape_reduction,
    minmax_augmentation_percentage  = minmax_augmentation_percentage,
    mask_vs_rotation_percentage = mask_vs_rotation_percentage,
    no_adjacent = no_adjacent
    )

parameters.add_modality(
    modality_name = "ADC", 
    batch_size=32, 
    #reshape_dim=(32,32,32),
    #reshape_dim=(32,128,96),
    reshape_dim=(32,32,32),
    skip_modality=True
    )
parameters.add_modality(
    modality_name = "t2tsetra", 
    #reshape_dim=None,  
    reshape_dim=(32,64,64),
    #reshape_dim=(32,128,96),
    batch_size=2,
    skip_modality=True
    )

parameters.join_modalities(["ADC", "t2tsetra"], decoder_method="upsample")
#parameters.join_modalities(["t2tsetra","ADC"])
    

y_train, y_val, pat_df = preprocess(parameters,True)



#parameters.set_current("Merged")


#asd=augment_patches(y_train[1][0:2])
#img_pltsave([a for a in asd])
#parameters.set_current("Merged")
# print(modality.image_shape)
# print((modality.image_shape[0]==modality.image_shape[1]))
# print(len(set(modality.image_shape))==1.)

#%%
print(modality.mrkdown())
#print(modality.input_name[1])


#[print(f"{modality.input_name[i]}") for i,y in enumerate(y_train)]
#[print(f"{modality.input_name[0]}") for i,y in enumerate(y_train)]

#modality.input_name[1]

#%%
models = {}
# for modality_name in parameters.lst:
#     parameters.set_current(modality_name)
import gc
for modality_name in parameters.lst.keys():
    parameters.set_current(modality_name)
    if modality.skip_modality:
        continue
    print(f"\nCurrent parameters:\n{modality.mrkdown()}")

    trainDS, valDS = augment_build_datasets(y_train[modality.idx], y_val[modality.idx])

    model = model_building(trainDS, valDS)
    
    models[modality_name] = model
    del model, trainDS, valDS
    tf.keras.backend.clear_session()
    gc.collect()

#%%
model = get_merged_model()
#model.summary()
#%%

# print(len(y_train[list(modality.idx[:])]))
# print(len(y_train[list(modality.idx)]))
# print(len(y_train[modality.idx]))
# print(f"\nCurrent parameters:\n{modality.mrkdown()}")
# print(isinstance(y_train[0], list))
# print(isinstance(y_train[modality.idx], np.ndarray))
# print(type(y_train[modality.idx]))
# print(type(y_train[0]))


#i = 0
# for yt,yv in (y_train,y_val):
#     if i == 0:
#         train_loader = tf.data.Dataset.from_tensor_slices((yt,yt))
#         val_loader   = tf.data.Dataset.from_tensor_slices((yv,yv))
#         i += 1
#     train_loader = train_loader.concatenate(tf.data.Dataset.from_tensor_slices((yt,yt)))
#     val_loader   = val_loader.concatenate(tf.data.Dataset.from_tensor_slices((yv,yv)))

isinstance(y_train, np.ndarray)

#%%

if isinstance(y_train, np.ndarray):


    train_loader = tf.data.Dataset.from_tensor_slices(({f"input_{i}":tf.repeat(augment_patches(y),3,-1) for i,y in enumerate(y_train)},{f"output_{i}":y for i,y in enumerate(y_train)}))

    val_loader = tf.data.Dataset.from_tensor_slices(({f"input_{i}":tf.repeat(augment_patches(y),3,-1) for i,y in enumerate(y_val)},{f"output_{i}":y for i,y in enumerate(y_val)}))


    PlotCallback.x_val = [augment_patches(y[0:modality.tensorboard_num_predictimages]) for y in y_val]

    trainDS = (
        train_loader
            .batch(
                batch_size = modality.batch_size
                ,num_parallel_calls=tf.data.AUTOTUNE)
            #.map(
                #lambda x, y: (tf.repeat(x,3,-1), y)
                #,num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(
                buffer_size = tf.data.AUTOTUNE)
                )
    valDS = (
                val_loader
                    .batch(
                        batch_size = modality.batch_size
                        ,num_parallel_calls=tf.data.AUTOTUNE)
                    # .map(
                    #     lambda x, y: (tf.repeat(x,3,-1), y)#tf.repeat(y,3,-1))
                    #     ,num_parallel_calls=tf.data.AUTOTUNE)
                    .prefetch(
                        buffer_size = tf.data.AUTOTUNE)
                        )
#%%

PlotCallback.x_val = [augment_patches(y[0:modality.tensorboard_num_predictimages]) for y in y_val]
#x_val = [augment_patches(y[0:modality.tensorboard_num_predictimages]) for y in y_val]


#%%

xv = [(y[0:modality.tensorboard_num_predictimages]) for y in y_val]
for k in range(len(xv[0])):
    img_pltsave([xv[i][k] for i in range(len(xv))])


#%%

#x_val = y_train[0]

#pred_sample = [np.zeros_like(x_val[i])  if len(x_val) > 1 else np.zeros_like(x_val) for i in range(len(x_val))]

pred_sample = [np.zeros_like(x_val[i]) for i in range(len(x_val))]
print(len(pred_sample))
print(pred_sample[0].shape)
print(pred_sample[1].shape)

#print(pred_sample.shape)
#%%

#x_val = [augment_patches(y[0:modality.tensorboard_num_predictimages]) for y in y_val]

asd = [[tf.repeat(tf.expand_dims(xval[i],0),3,-1) for xval in x_val] for i in range(len(x_val))]
print(len(asd))
print(asd[0].shape)
print(asd[1].shape)
#%%
pred_sample = [np.zeros_like(x_val[i]) for i in range(len(x_val))]
for i in range(modality.tensorboard_num_predictimages):
    #img = model.predict([tf.repeat(tf.expand_dims(xval[i],0),3,-1) for xval in x_val])
    #img = model.predict()
    for j in range(len(img)):
        pred_sample[j][i] = img[j]

for i, name in enumerate(modality.merged_modalities):        
    
    img_stack = np.stack([x_val[i],pred_sample[i]],0+1).reshape(*(-(0==j) or s for j,s in enumerate(pred_sample[i].shape)))
    image = img_pltsave([img for img in img_stack],None,True)
    with self.val_writer.as_default():
        tf.summary.image(modality.job_name+f"_img/{name}_{self.savename}", image, step=self.epoch, description=modality.mrkdown())
        self.val_writer.flush()


print(len(pred_sample))
print(pred_sample[0].shape)
print(pred_sample[1].shape)


#img_stack = np.stack([x_val,pred_sample],0+1).reshape(*(-(0==j) or s for j,s in enumerate(pred_sample.shape)))
#image = img_pltsave([img for img in img_stack])

#%%

#pred_sample[0].shape
len(pred_sample[0])


#%%
if isinstance(x_val, list):
        

    #x_val = y_train[0]

    #pred_sample = [np.zeros_like(x_val[i])  if len(x_val) > 1 else np.zeros_like(x_val) for i in range(len(x_val))]
    pred_sample = [np.zeros_like(x_val[i]) for i in range(len(x_val))]

    #for i in range(modality.tensorboard_num_predictimages):
    #    pred_sample[:][i] = model.predict([tf.repeat(tf.expand_dims(xval[i],0),3,-1) for xval in x_val])

    for i in range(2):#range(modality.tensorboard_num_predictimages):
        #qwe = model.predict([tf.repeat(tf.expand_dims(xval[i],0),3,-1) for xval in x_val])
        for k in range(len(pred_sample)):
            pred_sample[k][i] = qwe[k]

    for i, pred_sample in enumerate(pred_sample):

        img_stack = np.stack([x_val[i],pred_sample],0+1).reshape(*(-(0==j) or s for j,s in enumerate(pred_sample.shape)))

        image = img_pltsave([img for img in img_stack])


#%%
i = 0
qwe = model.predict([tf.repeat(tf.expand_dims(xval[i],0),3,-1) for xval in x_val])


#%%

# for xval in x_val:
#     print(xval[0].shape)

pred_sample = [np.zeros_like(x_val[i])  if len(x_val) > 1 else np.zeros_like(x_val) for i in range(len(x_val))]
#pred_sample = [np.zeros_like(x_val[i])  if len(x_val) > 1 else np.zeros_like(x_val) for i in range(len(x_val))]


print(pred_sample[0].shape)
print(y_val[0].shape)

#for i in range(modality.tensorboard_num_predictimages):
for i in range(2):
    for k in range(len(pred_sample)):
        pred_sample[k][i] = qwe[k]

for i, pred_sample in enumerate(pred_sample):

    img_stack = np.stack([x_val[i],pred_sample],0+1).reshape(*(-(0==j) or s for j,s in enumerate(pred_sample.shape)))

    image = img_pltsave([img for img in img_stack])
#with modality.strategy.scope():

# with self.val_writer.as_default():
#     tf.summary.image(modality.job_name+f"_img/{self.savename}", image, step=self.epoch, description=modality.mrkdown())
#     self.val_writer.flush()


#%%

print(len(qwe))
print(qwe[0].shape)
print(qwe[1].shape)
#%%
pred_sample = np.zeros_like(x_val)
for i,xval in enumerate(x_val):
#pred_sample = tf.concat([pred_sample,tf.repeat(self.model.predict(xval),3,-1)],0)
#pred_sample[:][i] = model.predict(tf.repeat(tf.expand_dims(xval,0),3,-1))
        
        img_stack = np.stack([self.__class__.x_val,pred_sample],0+1).reshape(*(-(0==j) or s for j,s in enumerate(pred_sample.shape)))

        image = img_pltsave([img for img in img_stack],None,True)
        #with modality.strategy.scope():
        with self.val_writer.as_default():
            tf.summary.image(modality.job_name+f"_img/{self.savename}", image, step=self.epoch, description=modality.mrkdown())
            self.val_writer.flush()
#%%
for i,xval in enumerate(x_val):
    #pred_sample = tf.concat([pred_sample,tf.repeat(self.model.predict(xval),3,-1)],0)
    #pred_sample[i] = self.model.predict(tf.repeat(tf.expand_dims(xval,0),3,-1))
    pred_sample[i] = model.predict(tf.repeat(tf.expand_dims(xval,0),3,-1))

img_stack = np.stack([x_val,pred_sample],0+1).reshape(*(-(0==j) or s for j,s in enumerate(pred_sample.shape)))

image = img_pltsave([img for img in img_stack])

#%%

[print(f"input_{i}: {y.shape}") for i,y in enumerate(y_train)]

#%%

# print(np.array(list(trainDS.as_numpy_iterator()))[0,0]["input_0"].shape)
# print(np.array(list(trainDS.as_numpy_iterator()))[0,1]["output_0"].shape)
# print(np.array(list(trainDS.as_numpy_iterator()))[0,0]["input_1"].shape)
# print(np.array(list(trainDS.as_numpy_iterator()))[0,1]["output_1"].shape)

print(np.array(list(trainDS.as_numpy_iterator()))[0,1]["output_1"].shape)
#%%

asd,qwe = next(iter(trainDS))


[print(k,v.shape) for k,v in asd.items()]
print()
#print(qwe)



#%%

importlib.reload(sys.modules['data_augmentation'])
importlib.reload(sys.modules['model_building'])
from data_augmentation import *
from model_building import *


models = {}
# for modality_name in parameters.lst:
#     parameters.set_current(modality_name)

for modality_name in parameters.lst.keys():
    parameters.set_current(modality_name)
    if modality.skip_modality:
        continue

    print(f"\nCurrent parameters:\n{modality.mrkdown()}")

    trainDS, valDS = augment_build_datasets(y_train[modality.idx], y_val[modality.idx])

    #model = model_building(trainDS, valDS)

    #asd = get_merged_model()
    #asd.summary()
    
    #models[modality_name] = model
    del model, trainDS, valDS
    tf.keras.backend.clear_session()
    gc.collect()

#%%
import segmentation_models_3D as sm
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.models import Model



def rename_all_layers(model, suffix):
    def _get_node_suffix(name):
        for old_name in old_nodes:
            if old_name.startswith(name):
                return old_name[len(name):]

    for i in range(len(model.layers)):
        layer = model.layers[i]
        old_name = layer.name
        new_name =  f'{old_name}_{suffix}'
        old_nodes = list(model._network_nodes)
        new_nodes = []
        for i,l in enumerate(model.layers):
            if l.name == old_name:
                l._name = new_name
                new_nodes.append(new_name + _get_node_suffix(old_name))
            else:
                new_nodes.append(l.name + _get_node_suffix(l.name))
        model._network_nodes = set(new_nodes)


def upsizedown(models_dict, method, half = None):

    models_dict = {key: (value if tf.is_tensor(value) else value.output) for key,value in models_dict.items()}
    out_sum = {key: sum(value.shape[1:-1]) for key,value in models_dict.items()}
    max_out, min_out = [m(out_sum, key = out_sum.get) for m in [max,min]]
    pad_tup, pool_tup =  [[],[]]
    def size_tuple(name):
        a,t = [enc.shape[1:-1] for enc in models_dict.values()]
        if half:
            idx = list(models_dict.keys()).index(name)
            if idx == 0:
                a = (max(1,x//2) for x in a)
            elif idx == 1:
                t = (max(1,x//2) for x in t)
        
        for a,t in zip(a,t):
            p = abs(a-t)
            
            #if p % 2 == 1 and p > 1:
            if p % 2 == 1:
                p_tup = (p//2,p//2+1)
            else:
                p_tup = (p//2,p//2)
            pad_tup.append(p_tup)
            pool_tup.append(max(a,t)//min(a,t))

    mod = {}
    ##UPSIZE##
    if method == "padd":
        #Padding
        size_tuple(max_out)
        mod[min_out] = tf.keras.layers.ZeroPadding3D(padding=(pad_tup),name=f'{min_out}_padding')(models_dict[min_out])
    elif method == "upsample":
        #UpSampling
        size_tuple(max_out)
        mod[min_out] = tf.keras.layers.UpSampling3D(size=(pool_tup),name=f'{min_out}_upsampling')(models_dict[min_out])
    elif method == "transpose":
        #Transpose
        size_tuple(max_out)
        pol2x = [x*2 for x in pool_tup]
        mod[min_out] = tf.keras.layers.Conv3DTranspose(512,pol2x,strides=pool_tup,padding="same",name=f'{min_out}_conv3dtrans')(models_dict[min_out])
        
    ##DOWNSIZE##
    elif method == "crop":
        #Cropping
        size_tuple(min_out)
        mod[max_out] = tf.keras.layers.Cropping3D(cropping=(pad_tup),name=f'{max_out}_cropping')(models_dict[max_out])
    elif method == "reshape":
        #Reshape
        size_tuple(min_out)
        mod[max_out] = tf.keras.layers.Reshape(target_shape=(*models_dict[min_out].shape[1:-1],-1), name=f'{max_out}_reshape')(models_dict[max_out])
    elif method == "maxpool":
        #MaxPooling
        size_tuple(min_out)
        mod[max_out] = tf.keras.layers.MaxPooling3D(pool_size=(pool_tup), strides = pool_tup,name=f'{max_out}_maxpooling')(models_dict[max_out])
    elif method == "avgpool":
        #AveragePooling
        size_tuple(min_out)
        mod[max_out] = tf.keras.layers.AveragePooling3D(pool_size=pool_tup, strides = pool_tup,name=f'{max_out}_averagepooling')(models_dict[max_out])
    
    return mod


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    #concat_axis = 4 if backend.image_data_format() == 'channels_last' else 1
    concat_axis = 4

    def wrapper(input_tensor, skip=None):

        if skip is not None:
            x_temp = layers.UpSampling3D(size=2, name=up_name)(input_tensor)
            if not np.array_equal(x_temp.shape[:-2] , skip.shape[:-2]):               
                xskip_dict = {'x':input_tensor,'skip':skip}
                mod = upsizedown(xskip_dict, modality.decoder_method)
                x,skip = [(value if key not in mod.keys() else mod[key]) for key,value in xskip_dict.items()]
            else:
                x = layers.UpSampling3D(size=2, name=up_name)(input_tensor)
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])
        else:
            x = layers.UpSampling3D(size=2, name=up_name)(input_tensor)
        x = sm.models.unet.Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = sm.models.unet.Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)
        return x
    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    #concat_axis = bn_axis = 4 if backend.image_data_format() == 'channels_last' else 1
    concat_axis = bn_axis = 4 

    def layer(input_tensor, skip=None):

        x_temp = layers.UpSampling3D(size=2, name=transp_name)(input_tensor)

        if skip is not None:
            if not np.array_equal(x_temp.shape[:-2] , skip.shape[:-2]):
                xskip_dict = {'x':input_tensor,'skip':skip}
                mod = upsizedown(xskip_dict, modality.decoder_method)
                x,skip = [(value if key not in mod.keys() else mod[key]) for key,value in xskip_dict.items()]
            else:
                x = layers.Conv3DTranspose(
                    filters,
                    kernel_size=(4, 4, 4),
                    strides=(2, 2, 2),
                    padding='same',
                    name=transp_name,
                    use_bias=not use_batchnorm,
                )(input_tensor)

                if use_batchnorm:
                    x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

                x = layers.Activation('relu', name=relu_name)(x)
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])
        else:
            x = layers.Conv3DTranspose(
                filters,
                kernel_size=(4, 4, 4),
                strides=(2, 2, 2),
                padding='same',
                name=transp_name,
                use_bias=not use_batchnorm,
            )(input_tensor)

            if use_batchnorm:
                x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

            x = layers.Activation('relu', name=relu_name)(x)

        x = sm.models.unet.Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)
        return x
    return layer


def build_merged_unet(
        backbone,
        decoder_block_type,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
        dropout=None,
):
    if decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2Block

    input_ = backbone.input    
    x = backbone.output

    skip_connection_layers = []
    for f in sm.Backbones.get_feature_layers(modality.backbone_name.lower(), n=4):
        skip_connection_layers += [f"{f}_{name}" for name in modality.merged_modalities]
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    if modality.backbone_name.startswith("vgg"):       
        x = sm.models.unet.Conv3x3BnReLU(modality.center_filter, use_batchnorm, name='center_block1')(x)
        x = sm.models.unet.Conv3x3BnReLU(modality.center_filter, use_batchnorm, name='center_block2')(x)

    for i in range(n_upsample_blocks):
        if modality.same_shape:
            skip = layers.Concatenate(axis=-1)([skips[i*2],skips[(i*2)+1]]) if i < len(skips)/2 else None
            x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)
        else:
            a,t = (a,t) if i != 0 else (x,x)
            adc, t2w = [skips[(i*2)+n] for n in range(2)] if i < len(skips)/2 else (None,None)
            a = decoder_block(decoder_filters[i], stage=f"{i}_a", use_batchnorm=use_batchnorm)(a, adc)
            t = decoder_block(decoder_filters[i], stage=f"{i}_t", use_batchnorm=use_batchnorm)(t, t2w)
        #elif mode == makes no sense
        #elif False:
            # if i < len(skips)/2:
        
            #     #skips_dict = {f"adc{i}":skips[i*2],f"t2w{i}":skips[(i*2)+1]}
            #     skips_dict = {f"adc{i}":adc,f"t2w{i}":t2w}
            #     mod = upsizedown(skips_dict, modality.encoder_method, half = False)
            #     t2w,adc = [(value if key not in mod.keys() else mod[key]) for key,value in skips_dict.items()]
            #     skip = layers.Concatenate(axis=-1)([adc,t2w])

            # x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)
            

    if dropout:
        x = layers.SpatialDropout3D(dropout, name='pyramid_dropout')(x)


    def final(x, name_suffix=""):
        x = layers.Conv3D(
            filters=classes,
            kernel_size=(3, 3, 3),
            padding='same',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name=f'final_conv_{name_suffix}',
        )(x)
        x = layers.Activation(activation, name=f"{activation}_{name_suffix}")(x)
        return x

    if modality.same_shape:
        x = final(x, name_suffix="")
        if modality.classes > 1:
            model = Model(input_, [tf.keras.layers.Lambda(tf.unstack, arguments=dict(axis=-1))(x)])
        else:
            model = Model(input_, x)
    else:
        a = final(a, name_suffix="_a")
        t = final(t, name_suffix="_t")
        model = Model(input_, [a,t])
        
    return model



def get_merged_model():
    encoders = {}
    for modality_name in modality.merged_modalities:

        dims = parameters.lst[modality_name]["image_shape"]
            
        model = sm.Unet(modality.backbone_name, input_shape=(dims[0],dims[1],dims[2],3), encoder_weights="imagenet", encoder_freeze = True)
        
        encoder = Model(model.input, model.get_layer(sm.Backbones.get_feature_layers(modality.backbone_name, n=1)[0]).output,name=f'encoder_{modality_name}')     

        rename_all_layers(encoder, modality_name)
        encoders[modality_name] = encoder

    if modality.same_shape:
        concat = layers.Concatenate(axis=-1)([encoders[key].output for key in encoders.keys()])
    else:
        out_sum = {key: sum(value.output.shape[1:-1]) for key,value in encoders.items()}
        max_out, min_out = [m(out_sum, key = out_sum.get) for m in [max,min]]
        if modality.encoder_method in ["upsample", "transpose", "padd"]:
            enc = max_out
        else: enc = min_out
        x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(encoders[enc].output)
        encoders[enc] = Model(encoders[enc].input, x)

        mod = upsizedown(encoders, modality.encoder_method)

        concat = layers.Concatenate(axis=-1)([(encoders[key].output if key not in mod.keys() else mod[key]) for key in encoders.keys()])

    model = keras.Model(inputs=[enc.input for enc in encoders.values()], outputs=[concat],name="concat_model")

    return build_merged_unet(
                backbone = model, 
                decoder_block_type = modality.decoder_block_type, 
                #decoder_filters=(1024,512,256, 128, 64),
                #decoder_filters=(512,256, 128, 64, 32), 
                #decoder_filters=(256, 128, 64, 32, 16), 
                decoder_filters = modality.decoder_filters
                n_upsample_blocks = 5, 
                classes = modality.classes, 
                activation = modality.activation, 
                use_batchnorm = True, 
                dropout = None
                )

asd = get_merged_model()

asd.summary()

tf.keras.utils.plot_model(
    asd,
    show_shapes=True,
    show_layer_activations = True,
    to_file=f'TEST_MERGE_{modality.modality_name}.png'
    #to_file=f'vgg16_t2w.png'
)


#%%
import segmentation_models_3D as sm

#bone = "vgg16"
bone = "resnet18"

#model = sm.Unet(bone, input_shape=(32,384,384,3), encoder_weights="imagenet", encoder_freeze = True)
#model = sm.Unet(bone, input_shape=(32,128,96,3), encoder_weights="imagenet", encoder_freeze = True)
model = sm.Unet(bone, input_shape=(32,128,96,3), encoder_weights="imagenet", encoder_freeze = True)

#model = sm.Unet(bone, input_shape=(32,128,96,3), encoder_weights="imagenet", encoder_freeze = True, decoder_block_type="transpose")
#model = sm.Unet(bone, input_shape=(32,128,96,3), encoder_weights="imagenet", encoder_freeze = True, decoder_block_type="transpose")

#model = sm.Unet(bone, input_shape=(None,None,None,3), encoder_weights="imagenet", encoder_freeze = True)

# from keras.layers import Input, Conv3D
# from keras.models import Model
#from tensorflow.keras.layers import Input, Conv3D
#from tensorflow.keras.models import Model
#%%

from tensorflow.keras.layers import Input, Conv3D
from tensorflow.keras import Model
inp = Input(shape=(None, None, None, 1))
model = sm.Unet(bone, input_shape=(None, None, None,3), encoder_weights="imagenet", encoder_freeze = True)
#inp = Input(shape=(32,128,96, 1))
l1 = Conv3D(3, (1, 1, 1))(inp) # map N channels data to 3 channels
out = model(l1)

model = Model(inp, model(l1), name=model.name)
model.summary()

#%%
#model.get_layer(index=-1).summary()
model = sm.Unet(bone, input_shape=(None, None, None,3), encoder_weights="imagenet", encoder_freeze = True)
model.summary()

#sm.Backbones.get_feature_layers(bone, n=1)[0]

#%%
import tensorflow as tf
tf.keras.utils.plot_model(
    model,
    show_shapes=True,
    show_layer_activations = True,
    to_file=f'{bone}.png'
)
#%%
sm.Backbones.get_feature_layers(bone, n=1)[0]

#%%

ind = [layer.name for layer in model.layers].index("decoder_stage0_upsampling")-1
print(ind)
print(model.layers[ind].name)

#skip_connection_layers = []
#for f in sm.Backbones.get_feature_layers(modality.backbone_name.lower(), n=4):
#        skip_connection_layers += [f"{f}_{name}" for name in modality.merged_modalities]

#%%

#bone = "vgg16"
#bone = "resnet18"
#model = sm.Unet(bone, input_shape=(32,384,384,3), encoder_weights="imagenet", encoder_freeze = True)
#model.summary()
sm.Backbones.get_feature_layers(bone, n=1)[0]
#%%
out_sum = [sum(enc_adc.output.shape[1:]),sum(enc_t2w.output.shape[1:])]
largest_index = out_sum.index(max(out_sum))
print(asd)
print(sum(enc_adc.output.shape[1:]))
print(sum(enc_t2w.output.shape[1:]))
#%%

#x = 3+2
#print((12-3)/2)
#print(enc_adc.output.shape)
pad_tup = []
for a,t in zip((enc_t2w.output.shape[1:-1]),(enc_adc.output.shape[1:-1])):
    p = abs(t-a)
    print(p)
    if p % 2 == 1:
        p_tup = (p//2,p//2+1)
    else:
        p_tup = (p//2,p//2)
    pad_tup.append(p_tup)

enc_adc_pad = tf.keras.layers.ZeroPadding3D(padding=(pad_tup))(enc_adc.output)

print(pad_tup)

print(enc_adc.output.shape)
enc_adc_pad.shape
#%%

#parameters.set_current("Merged")
(modality.image_shape[0]==modality.image_shape[1])

#%%
pred_adc = tf.repeat(augment_patches(y_val[0][0]),3,-1)
pred_t2w = tf.repeat(augment_patches(y_val[1][0]),3,-1)

print("pred_adc.shape",pred_adc.shape)
print("pred_t2w.shape",pred_t2w.shape)
#%%
# predictions = asd.predict([pred_adc,pred_t2w])
# print("predictions.shape", predictions.shape)

pred_a, pred_t = asd.predict([pred_adc,pred_t2w])
print("pred_a.shape", pred_a.shape)
print("pred_t.shape", pred_t.shape)

# try:
#     predictions = asd.predict([pred_adc,pred_t2w])
#     print("predictions.shape", predictions.shape)
# except Exception as first:
#     try:
#         pred_a, pred_b = asd.predict([pred_adc,pred_t2w])
#         print("pred_a.shape", pred_a.shape)
#         print("pred_b.shape", pred_b.shape)
#     except Exception as second:
#         print("first", first)
#         print("*"*50)
#         print("second", second)
#         raise Exception("Something went wrong")

#%%
img_pltsave(pred_a[0])
img_pltsave(pred_t[0])

#%%


# def rename_layer(model, layer, new_name):
#     def _get_node_suffix(name):
#         for old_name in old_nodes:
#             if old_name.startswith(name):
#                 return old_name[len(name):]

#     old_name = layer.name
#     old_nodes = list(model._network_nodes)
#     new_nodes = []

#     for l in model.layers:
#         # if l.name == "center_block2_relu":
#             #continue
#         if l.name == old_name:
#             l._name = new_name
#             new_nodes.append(new_name + _get_node_suffix(old_name))
#         else:
#             new_nodes.append(l.name + _get_node_suffix(l.name))
#     model._network_nodes = set(new_nodes)

def rename_all_layers(model, suffix):
    def _get_node_suffix(name):
        for old_name in old_nodes:
            if old_name.startswith(name):
                return old_name[len(name):]

    
    
    idx_center_block2_relu = 0
    idx_center_block1_conv = 0
    for i in range(len(model.layers)):
        layer = model.layers[i]
        old_name = layer.name
        new_name =  f'{old_name}_{suffix}'
        old_nodes = list(model._network_nodes)
        new_nodes = []
        for i,l in enumerate(model.layers):
            if l.name == "center_block1_conv":
                idx_center_block1_conv = i
            if l.name == "center_block2_relu":
                idx_center_block2_relu = i



            if l.name == old_name:
                l._name = new_name
                new_nodes.append(new_name + _get_node_suffix(old_name))
            else:
                new_nodes.append(l.name + _get_node_suffix(l.name))
        model._network_nodes = set(new_nodes)

    print("idx_center_block2_relu ",idx_center_block2_relu)
    print("idx_center_block1_conv ",idx_center_block1_conv)

models = {}

for modality_name in parameters.tags.keys():
    print(modality_name)
    # modelpath = f"../models/{modality}/{os.environ['SLURM_JOB_NAME']}/{os.environ['SLURM_JOB_ID']}-{os.environ['SLURM_JOB_NAME']}"
    #modelpath = f"../models/{modality}/{os.environ['SLURM_JOB_NAME']}"

    shape,idx = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality_name, case=False)].values[0]
    # train_data = tf.data.Dataset.from_tensor_slices((x_train_noisy[idx], x_train[idx]))
    # val_data = tf.data.Dataset.from_tensor_slices((x_test_noisy[idx], x_test[idx]))
    # models[modality] = model_building(shape, modelpath, train_data, val_data)
    #https://stackoverflow.com/questions/52724022/model-fits-on-a-single-gpu-but-script-crashes-when-trying-to-fit-on-multiple-gpu

    #model = model_building(shape, modelpath, x_train[idx],y_train[idx], x_test[idx], y_val[idx])
    

    #trainDS, valDS = augment_build_datasets(y_train[idx], y_val[idx])

    #model = model_building(shape, f"{modality}", trainDS, valDS)
    dim = shape
    #model = sm.Unet("vgg16", input_shape=(dim[0], dim[1], dim[2],3), encoder_weights="imagenet")
    model = sm.Unet("vgg16", input_shape=(None,None,None,3), encoder_weights="imagenet", encoder_freeze = True)


    # for i in range(len(models[modality].layers)):
    #     rename_layer(models[modality], models[modality].layers[i], f'{models[modality].layers[i]._name}_{modality}')

    # for i in range(len(model.layers)):
    #     rename_layer(model, model.layers[i], f'{model.layers[i]._name}_{modality}')

    
    rename_all_layers(model, modality_name)
    models[modality_name] = model
    #models.append(model)
    # del model
    # del trainDS
    # del valDS
    tf.keras.backend.clear_session()
    gc.collect()



#%%


import segmentation_models_3D as sm
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D



#enc_adc = Model(models["ADC"].input, models["ADC"].get_layer('center_block2_relu_ADC').output,name='enc_adc')
enc_adc = Model(models["ADC"].input, models["ADC"].get_layer('block2_pool_ADC').output,name='enc_adc')

#enc_t2w = Model(models["t2tsetra"].input, models["t2tsetra"].get_layer('center_block2_relu_t2tsetra').output,name='enc_t2w')
enc_t2w = Model(models["t2tsetra"].input, models["t2tsetra"].get_layer('block2_pool_t2tsetra').output,name='enc_t2w')

#test = tf.keras.layers.ZeroPadding3D(padding=(0, 4,4))(enc_adc.output)
#test = tf.keras.layers.ZeroPadding3D(padding=((0,0),(4,4),(4,5)))(enc_adc.output)

#test = tf.keras.layers.ZeroPadding3D(padding=4)(enc_adc.output)

#test.shape

#concat =  concatenate([test, enc_t2w.output])
concat = tf.keras.layers.Concatenate(axis=-1)([enc_adc.output, enc_t2w.output])
#concat = tf.keras.layers.Concatenate(axis=-1)([test, enc_t2w.output])
#concat =  concatenate([tf.keras.layers.Flatten()(enc_adc.output), tf.keras.layers.Flatten()(enc_t2w.output)],name="concat")

#TODO: Mulig det går ann å bruke Conv3D for å reshape XYZ å kjøre resten i channel ? siden channel må være lik

# output = tf.keras.layers.Dense(1, name="out")(concat)

model = keras.Model(inputs=[enc_adc.input, enc_t2w.input], outputs=[concat],name="asd")#, axis=-1)

model.summary()



#%%


#pred_adc = tf.repeat(augment_patches(y_val[0]),3,-1)
#predictions_adc = enc_adc.predict(pred_adc)
##adc_input = tf.keras.Input(shape=(predictions_adc.shape))
adc = tf.repeat(augment_patches(y_val[0][0]),3,-1)

predictions_adc = enc_adc(adc)
#adc_input = predictions_adc.flatten()
adc_input = tf.keras.Input(shape=(predictions_adc.shape))


#pred_t2w = tf.repeat(augment_patches(y_val[1]),3,-1)
#predictions_t2w = enc_t2w.predict(pred_t2w)
##t2w_input = tf.keras.Input(shape=(predictions_t2w.shape))

t2w = tf.repeat(augment_patches(y_val[1][0]),3,-1)
predictions_t2w = enc_t2w(t2w)
#t2w_input = predictions_t2w.flatten()
t2w_input = tf.keras.Input(shape=(predictions_t2w.shape))

#tf.keras.layers.concatenate([predictions_adc,predictions_t2w],axis=-1)
concat = tf.keras.layers.concatenate([adc_input,t2w_input],axis=-1)

t2w_input = tf.keras.Input(shape=(predictions_t2w.shape))

dec_adc = Model(concat, models["ADC"].layers[-1](concat))
dec_adc.summary()

#%%


dec_adc = Model(adc_input, models["ADC"].layers[-1](adc_input))

test = Model(enc_adc,dec_adc)
test.summary()


#%%

test = Model(concat, models["ADC"].layers[-1](concat))
test.summary()
#%%
#[layer.name for layer in models["ADC"].layers]

predictions = model([adc,t2w])

print(predictions.shape)
all_input = tf.keras.Input(shape=(predictions.shape))
print(all_input.shape)


dec_adc = Model(concat, [models["ADC"].layers[-1]](concat))
dec_adc.summary()



#%%

layer_outputs = []
decoder = Sequential()
for i in range(24, len(models["ADC"].layers)):
    tmp_model = Model(models["ADC"].layers[0].input, models["ADC"].layers[i].output)
    tmp_output = tmp_model.predict(adc)#[0]
    layer_outputs.append(tmp_output)
    


#%%

models["ADC"].summary()
#%%


import segmentation_models_3D as sm
from segmentation_models_3D.models import *
#unet
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D,InputLayer, Flatten

tf.keras.backend.clear_session()

all_input = tf.keras.Input(shape=(None, None, None, 3), name="input_1_ADC_")


#model_test = sm.Unet("vgg16", encoder_weights="imagenet", input_shape=(None, None, None, 3),encoder_freeze=True)

#model_test.summary()

#sm.models.unet.get_submodules()

sm.backbones.backbones_factory.BackbonesFactory().get_backbone(name="vgg16", input=(None, None, None, 3), encoder_weights="imagenet", encoder_freeze=True)
#VGG16(include_top=False, input_shape=(None, None, None, 3))


#enc_adc = Model(models["ADC"].input, models["ADC"].get_layer('block2_pool_ADC').output,name='enc_adc')
#enc_adc = Model(models["ADC"].input, models["ADC"].layers[18].output,name='enc_adc')
#print(enc_adc.summary())

#%%
#sm.Backbones.get_backbone("vgg16",input_shape=(None, None, None, 3), weights="imagenet", include_top=True)


import segmentation_models_3D as sm
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D

def build_unet_test(
        backbone,
        decoder_block_type,
        skip_connection_layers,
        #decoder_filters=(1024,512,256, 128, 64, 32, 16),
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
        dropout=None,
):
    if decoder_block_type == 'upsampling':
        decoder_block = sm.models.unet.DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = sm.models.unet.DecoderTransposeX2Block

    input_ = backbone.input    
    x = backbone.output
    

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    #if isinstance(backbone.layers[-1], layers.MaxPooling3D):
    if isinstance(backbone.layers[-1], tensorflow.keras.layers.Concatenate):
        x = sm.models.unet.Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = sm.models.unet.Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)
        #x = sm.models.unet.Conv3x3BnReLU(1024, use_batchnorm, name='center_block1')(x)
        #x = sm.models.unet.Conv3x3BnReLU(1024, use_batchnorm, name='center_block2')(x)

    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips)/2:
            #skip = [skip_connections[i*2],skip_connections[(i*2)+1]]
            skip = tensorflow.keras.layers.Concatenate(axis=-1)([skips[i*2],skips[(i*2)+1]])
        else:
            skip = None
        
    x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)
        

    if dropout:
        x = tensorflow.keras.layers.SpatialDropout3D(dropout, name='pyramid_dropout')(x)

    # model head (define number of output classes)
    x = tensorflow.keras.layers.Conv3D(
        filters=classes,
        kernel_size=(3, 3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = tensorflow.keras.layers.Activation(activation, name=activation)(x)

    model = Model(input_, x)
    return model




enc_adc = Model(models["ADC"].input, models["ADC"].get_layer('block5_pool_ADC').output,name='enc_adc')
enc_t2w = Model(models["t2tsetra"].input, models["t2tsetra"].get_layer('block5_pool_t2tsetra').output,name='enc_t2w')
concat = tf.keras.layers.Concatenate(axis=-1)([enc_adc.output, enc_t2w.output])
model = keras.Model(inputs=[enc_adc.input, enc_t2w.input], outputs=[concat],name="asd")



skip_connections = ()
for f in sm.Backbones.get_feature_layers(modality.backbone_name.lower(), n=4):
    skip_connections += (f"{f}_ADC",f"{f}_t2tsetra")

asd = build_unet_test(
                backbone = model, 
                decoder_block_type = modality.decoder_block_type, 
                skip_connection_layers = skip_connections, 
                decoder_filters = (256, 128, 64, 32, 16), 
                n_upsample_blocks = 5, 
                classes = 1, 
                activation = 'sigmoid', 
                use_batchnorm = True, 
                dropout = None
                )



#pred_adc = tf.repeat(augment_patches(y_val[0][0]),3,-1)
#pred_t2w = tf.repeat(augment_patches(y_val[1][0]),3,-1)
predictions = asd.predict([pred_adc,pred_t2w])


#%%
print(modality.mrkdown())
#%%
skip_connections = ()
for i in sm.Backbones.get_feature_layers(backbone.lower(), n=4):
    #print(i)
    skip_connections += (f"{i}_ADC",f"{i}_t2tsetra")
print(swips)
#%%
#model_test = sm.Unet("vgg16", encoder_weights="imagenet", input_shape=(None, None, None, 3),encoder_freeze=True)
#sm.Backbones.get_backbone("vgg16",input_shape=(None, None, None, 3), weights="imagenet", include_top=False)

all_input = tf.keras.Input(shape=( None, None, None, 512), name="input_1")
asd_model = Model(inputs=all_input,outputs=asd)
#%%



#for i,lay in enumerate(model_test.layers[7:]):
for i,lay in enumerate(model_test.layers[:]):

    #print(i, models["ADC"].layers[i].name)
    try:
        #qwe = model_test.layers[i].input #.get_layer('block5_pool').output
        qwe = lay.input

        #all_input = tf.keras.Input(shape=(None, None, None, None, 512), name="input_1")
        asd_model = Model(inputs=qwe,outputs=asd)
        print(i,lay.name)
    except Exception as e:
        #print(e)
        #print(i,lay.name)
        pass
        #break

#asd_model.summary()


#%%

all_input = tf.keras.Input(shape=( None, None, None, 512), name="input_1")
asd_model = Model(inputs=all_input,outputs=asd)
#%%
#print(model_test.get_layer('block1_conv2').output)
backbone = "vgg16"

fl_names = sm.Backbones.get_feature_layers(backbone.lower())
for name in fl_names:
    print(name)
    #print(model_test.get_layer(name).output)

print(sm.Backbones.get_feature_layers(backbone.lower(), n=1))
#enc_adc = Model(model_test.input, model_test.get_layer('block5_pool').output,name='enc_adc')
#enc_adc.summary()

#%%


encoder = Sequential()
#encoder.add(all_input)
#encoder.add(InputLayer(input_shape = (None, None, None, 3)))
decoder = Sequential()
#for i,lay in enumerate(models["ADC"].layers):
for i,lay in enumerate(model_test.layers):

    #print(i, models["ADC"].layers[i].name)
    try:
        #Model(models["ADC"].layers[i].input, models["ADC"].output, name="decoder")
    #    print(i, models["ADC"].layers[i].name)
    
        # #if not any(lay.name.startswith(s) for s in ["block","input","center","final","sigmoid"]):
        # if not any(lay.name.startswith(s) for s in ["block","input","final","sigmoid"]):
        #     decoder.add(models["ADC"].get_layer(name=lay.name))
        # elif not any(lay.name.startswith(s) for s in ["input","center","decoder","final","sigmoid"]):
        #     encoder.add(models["ADC"].get_layer(name=lay.name))

        #if not any(lay.name.startswith(s) for s in ["block","input","center","final","sigmoid"]):
        if not any(lay.name.startswith(s) for s in ["block","input"]):
            decoder.add(model_test.get_layer(name=lay.name).output)
            print(i)
        #elif not any(lay.name.startswith(s) for s in ["input","center","decoder","final","sigmoid"]):
        elif not any(lay.name.startswith(s) for s in ["input","center","decoder","final","sigmoid"]):
            encoder.add(model_test.get_layer(name=lay.name))
        
    except:
        #pass
        
        #encoder.add(models["ADC"].get_layer(name=lay.name))
        print("Error: ",i, model_test.layers[i].name)


#model = keras.models.Model(inputs=[ all_input ], outputs = [encoder] )
#model.summary()
#encoder.build(input_shape=(None, None, None, 3))
#asd = InputLayer(input_shape=(None,None, None, None, 3))
asd = Input(shape=(None, None, None, 3))
#encoder = Model(inputs=[asd], outputs=encoder(asd))
encoder = Model(inputs=asd, outputs=encoder(asd))
#encoder.build(input_shape=(None, None, None, None,3))
#encoder.summary()

#asd = Input(shape=(None, None, None, 3))
decoder = Model(inputs=encoder.output, outputs=decoder(encoder.output))
decoder.summary()
#decoder(encoder.output)
#decoder.build(encoder.output)
#decoder.summary()
#%%
#predictions_adc
#adc_input
all_input = tf.keras.Input(shape=(None, None, None, None, 512), name="input_1_ADC")
Model(models["ADC"].layers[25](all_input), models["ADC"].layers[-1](all_input), name="decoder")
#%%
# extract decoder fitted weights
img_shape = (200,200,1)


last_encoder_layer = 18

restored_w = []
for w in models["ADC"].layers[last_encoder_layer + 1:]:
    restored_w.extend(w.get_weights())
  
# reconstruct decoder architecture setting the fitted weights
new_inp = [Input(l.shape[1:]) for l in enc_adc]#get_encoder(img_shape)]
new_dec = get_decoder(new_inp)
decoder = Model(new_inp, new_dec)
decoder.set_weights(restored_w)

decoder.summary()

#%%

backbone = "vgg16"
framework = sm.Unet(backend=backbone,  
                    encoder_weights='imagenet',
                    activation="sigmoid",
                    classes=1,
                    input_shape=(None,None,None, 3))

#%%
encoder_features = sm.Backbones.get_feature_layers(backbone.lower(), n=100)
final_downsampling_feature = framework.get_layer(name=encoder_features[10]).output
model = keras.models.Model(framework.input, final_downsampling_feature)
model.summary()

#%%
#sm.Backbones.get_feature_layers("vgg16")
decoder = Sequential()
decoder.add(framework.get_layer(name="decoder_stage0_upsampling"))
decoder.add(framework.get_layer(name="decoder_stage0_concat"))
decoder.add(framework.get_layer(name="decoder_stage0a_conv"))
decoder.add(framework.get_layer(name="decoder_stage0a_bn"))

#all_input = tf.keras.Input(shape=(None, None, None, None, 512), name="input_1_ADC")
#all_input = tf.keras.Input(shape=(512, None, None, None, None), name="input_1_ADC")
decoder.build(enc_adc.output)
decoder.summary()
#%%

bottleneck_index = 24 # this you need to identify 19 24
ae_model = models["ADC"]
encoder_model = tf.keras.Sequential()
for i,layer in enumerate(ae_model.layers[bottleneck_index:-1]):
    layer_config = layer.get_config()  # to get all layer's parameters (units, activation, etc...)
    copied_layer = type(layer).from_config(layer_config) # to initialize the same layer class with same parameters
    #print(i)
    #print(layer.input_shape)
    
    idx = bottleneck_index + i 
    
    #print(ae_model.get_input_shape_at(idx))
    copied_layer.build(layer.input_shape)  # build the layer to initialize the weights.
    
    #print(layer.get_input_shape_at(idx))
    #copied_layer.build(layer.get_input_shape_at(idx))  # build the layer to initialize the weights.
    copied_layer.set_weights(layer.get_weights())  # transfer the trainable parameters
    encoder_model.add(copied_layer)  # add it to the encoder's model

#encoder_model.build(ae_model.get_input_shape_at(bottleneck_index))
encoder_model.build(input_shape=(None, None, None, None, 512))
encoder_model.summary()

#%%

model = Model(inputs=models["ADC"].layers[24].input, outputs=models["ADC"].output, name="decoder")
model.summary()


#%%

ValueError: Found input tensor cannot be reached given provided output tensors. Please make sure the tensor KerasTensor(type_spec=TensorSpec(shape=(None, None, None, None, 3), dtype=tf.float32, name='input_1'), name='input_1', description="created by layer 'input_1_retest'") is included in the model inputs when building functional model.
#%%
decoder = Sequential()
#for i in range(7,12):
for i in range(24, len(models["ADC"].layers)):
    decoder.add(models["ADC"].layers[i])

decoder.build(input_shape=(None, None, None, None, 512))
decoder.summary()

#%%
for i in layer_outputs:
    print(i.shape)
#layer_outputs[0].shape
#%%

train_loader = tf.data.Dataset.from_tensor_slices((tf.repeat(augment_patches(y_train[0]),3,-1), y_train[0]))
val_loader = tf.data.Dataset.from_tensor_slices((tf.repeat(augment_patches(y_val[0]),3,-1), y_val[0]))


#%%

import segmentation_models_3D as sm
dim = (2, 24, 24, 512)
asd = sm.Unet("vgg16", input_shape=(dim[0], dim[1], dim[2],dim[3]), classes=1, encoder_weights=None,encoder_features=["block5_conv3"])
#dim = (32,384,384,1)
#asd = sm.Unet("vgg16", input_shape=(dim[0], dim[1], dim[2],3), classes=1, encoder_weights="imagenet")
#asd = sm.Unet("vgg16", input_shape=(None,None,None,3),classes= 1, encoder_weights="imagenet",encoder_features=["block5_pool"])
asd.summary()

#%%

import segmentation_models_3D as sm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D

dim = (32,32,32,1)
#dim = (96,256,256)
asd = sm.Unet("vgg16", input_shape=(dim[0], dim[1], dim[2],3), classes=1, encoder_weights="imagenet")
#asd = sm.Unet("vgg16", classes= 1, encoder_weights="imagenet")
N = 1#dim.shape[-1]
#inp = Input(shape=(None, None,None, N))
inp = Input(shape=(dim[0], dim[1], dim[2],1))
l1 = Conv3D(3, (1,1,1))(inp) # map N channels data to 3 channels
out = asd(l1)

asd = Model(inp, out, name=asd.name)

asd.summary()

[layer.name for layer in asd.layers]
#%%

BACKBONE = 'vgg16'
preprocess_input = sm.get_preprocessing(BACKBONE)
asd = preprocess_input(tf.repeat(augment_patches(y_val[0][0:5]),3,-1))
#%%
[layer.name for layer in models["ADC"].layers]

#%%

pred_test = tf.repeat(augment_patches(y_val[modality.idx]),3,-1)

print("shape staert")
print(pred_test.shape)
print("prediction")
predictions = models["ADC"].predict(pred_test)
print(predictions.shape)
print("prediction")
img_pltsave([y_val[modality.idx][0], pred_test, predictions])

#%%


#%%
encoder_features =  ('block5_conv3_ADC', 'block4_conv3_ADC', 'block3_conv3_ADC', 'block2_conv2_ADC', 'block1_conv2_ADC'
                    ,'block5_conv3_t2tsetra', 'block4_conv3_t2tsetra', 'block3_conv3_t2tsetra', 'block2_conv2_t2tsetra', 'block1_conv2_t2tsetra')

from keras.models import Model
from keras.layers import concatenate, Input, Conv3D,  Conv3DTranspose
filterFactor=1


conv4 = Conv3D(filterFactor, (3, 3, 3), activation='relu', padding='same')(concat)

# conv3_tra = enc_adc.get_layer('block5_conv3_ADC').input
# conv3_cor = enc_t2w.get_layer('block3_conv3_t2tsetra').input
conv3_tra = enc_adc.output
conv3_cor = enc_t2w.output
# conv3_tra = enc_adc.get_layer('block3_conv3_ADC').output
# conv3_cor = enc_t2w.get_layer('block3_conv3_t2tsetra').output

up6 = Conv3DTranspose(256,(2,2,2), strides = (2,2,2), activation = 'relu', padding = 'same' )(conv4)

up6 = tf.keras.layers.Concatenate(axis=-1)([up6, conv3_tra, conv3_cor])
conv6 = Conv3D(256*filterFactor, (3, 3, 3), activation='relu', padding='same')(up6)
conv6 = Conv3D(256*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv6)

conv2_tra = enc_adc.get_layer('block2_conv2_ADC')
conv2_cor = enc_t2w.get_layer('block2_conv2_t2tsetra')

up7 = Conv3DTranspose(256,(2,2,2), strides = (2,2,2), activation = 'relu', padding = 'same' )(conv6)
up7 = concatenate([up7, conv2_tra, conv2_cor])
conv7 = Conv3D(128*filterFactor, (3, 3, 3), activation='relu', padding='same')(up7)
conv7 = Conv3D(128*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv7)

conv1_tra = enc_adc.get_layer('block1_conv2_ADC')
conv1_cor = enc_t2w.get_layer('block1_conv2_t2tsetra')

up8 = Conv3DTranspose(128,(2,2,2), strides = (2,2,2), activation = 'relu', padding = 'same' )(conv7)
up8 = concatenate([up8, conv1_tra, conv1_cor])
conv8 = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(up8)
conv8 = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv8)

conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv8)

inputs_tra = enc_adc.input.shape[1:]
inputs_cor = enc_t2w.input.shape[1:]

model = Model(inputs=[inputs_tra, inputs_cor], outputs=[conv10])


#%%

#isinstance(enc_adc.layers[-1], layers.MaxPooling3D)
#enc_adc.layers[-1]

tf.keras.layers.MaxPooling3D
#tf.keras.layers.MaxPool3D
model.output.shape
#%%

#import importlib
#importlib.reload(sys.modules['segmentation_models_3D.models.unet'])
from segmentation_models_3D.models.unet  import *

backbone = model
backbone_name = "vgg16"
input_shape=model.output.shape#(32,128,96,3)#(None, None, 3),
#enc_adc.output.shape
classes=1
activation='sigmoid'
weights=None
encoder_weights='imagenet'
encoder_freeze=False
#encoder_features='default'
decoder_block_type='upsampling'
decoder_filters=(256, 128, 64, 32, 16)
decoder_use_batchnorm=True
dropout=None


decoder_block = DecoderUpsamplingX2Block

# if decoder_block_type == 'upsampling':
#     decoder_block = DecoderUpsamplingX2Block
# elif decoder_block_type == 'transpose':
#     decoder_block = DecoderTransposeX2Block
# else:
#     raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
#                         'Got: {}'.format(decoder_block_type))

#if encoder_features == 'default':
        #encoder_features = Backbones.get_feature_layers(backbone_name, n=4)
encoder_features =  ('block5_conv3_ADC', 'block4_conv3_ADC', 'block3_conv3_ADC', 'block2_conv2_ADC', 'block1_conv2_ADC'
                    ,'block5_conv3_t2tsetra', 'block4_conv3_t2tsetra', 'block3_conv3_t2tsetra', 'block2_conv2_t2tsetra', 'block1_conv2_t2tsetra')

model = build_unet(
    backbone=backbone,
    decoder_block=decoder_block,
    skip_connection_layers=encoder_features,
    decoder_filters=decoder_filters,
    classes=classes,
    activation=activation,
    n_upsample_blocks=len(decoder_filters),
    use_batchnorm=decoder_use_batchnorm,
    dropout=dropout,
)

model.summary()


#%%

Backbones.get_feature_layers(backbone_name, n=4)

#%%

bn_axis = 4

x = layers.BatchNormalization(axis=bn_axis, name="bn_name")(x)

#x = layers.Activation(activation, name=act_name)(x)

#%%


#decoder_stage0_upsampling_t2tsetra,block5_conv3_t2tsetra

inshape = tf.keras.Input(shape=(concat.shape[1:]),name="inshape")
#enc_t2w = Model([models["t2tsetra"].get_layer('decoder_stage0_upsampling_t2tsetra').output,models["t2tsetra"].get_layer('block5_conv3_t2tsetra').output], models["t2tsetra"].layers[-1].output,name='enc_t2w')(inputs)

# enc_t2w.summary()

#enc_t2w = Model(inputs,models["t2tsetra"].get_layer('decoder_stage0_upsampling_t2tsetra')(inputs),name='enc_t2w')
enc_t2w = Model(inputs=models["t2tsetra"].get_layer('decoder_stage0_upsampling_t2tsetra')(inputs),outputs=models["t2tsetra"].layers[-1],name='enc_t2w')

#enc_t2w = Model(inputs, models["t2tsetra"].get_layer('center_block2_relu_t2tsetra')(inputs))
#enc_t2w.build(input_shape=concat.shape)
enc_t2w.summary()
#%%

#models["t2tsetra"].summary()

"""
(None, 1, 4, 3, 512) 0 - center_block2_relu_ADC
(None, 1, 12, 12, 512) 0 - center_block2_relu_t2setra

(None, 2, 8, 6, 512) 0 - decoder_stage0_upsampling_ADC
(None, 2, 24, 24, 512) 0 - decoder_stage0_upsampling_t2tsetra

"""
#dec_adc = Model(models["ADC"].get_layer('center_block2_relu').input,models["ADC"].layers[-1].output,name='enc_adc')
dec_adc = Model(output,models["ADC"].layers[-1].output,name='enc_adc')
#dec_t2w = Model(models["t2tsetra"].input, models["t2tsetra"].get_layer('center_block2_relu').output,name='enc_t2w')



#%%

# adc = tf.repeat(augment_patches(y_val[0]),3,-1)
# predictions_adc = enc_adc(adc)

# test = tf.repeat(augment_patches(y_val[0]),3,-1)
#test_pred = model.predict([adc,t2w])

test_pred.shape

test_pred[0].shape

#%%

# decoder_layer = models["ADC"].layers[-1]
# dec_adc = Model(enc_adc, decoder_layer(enc_adc))

#dec_adc = Model(models["ADC"].get_layer('center_block2_relu').input, models["ADC"].layers[-1])
#dec_adc = Model(enc_adc, models["ADC"].layers[-1].output)

# dec_adc.summary()
# opt = tf.keras.optimizers.Adam()
# learning_rate = opt.lr.numpy()*len(tf.config.list_physical_devices('GPU'))
# opt.lr.assign(learning_rate)
# dec_adc.compile(opt, loss="binary_crossentropy")


#pred_adc = tf.repeat(augment_patches(y_val[0]),3,-1)
#predictions_adc = enc_adc.predict(pred_adc)
##adc_input = tf.keras.Input(shape=(predictions_adc.shape))
adc = tf.repeat(augment_patches(y_val[0][0]),3,-1)
predictions_adc = enc_adc(adc)
#adc_input = predictions_adc.flatten()
adc_input = tf.keras.Input(shape=(predictions_adc.shape))

#pred_t2w = tf.repeat(augment_patches(y_val[1]),3,-1)
#predictions_t2w = enc_t2w.predict(pred_t2w)
##t2w_input = tf.keras.Input(shape=(predictions_t2w.shape))

t2w = tf.repeat(augment_patches(y_val[1][0]),3,-1)
predictions_t2w = enc_t2w(t2w)
#t2w_input = predictions_t2w.flatten()
t2w_input = tf.keras.Input(shape=(predictions_t2w.shape))


#tf.keras.layers.concatenate([predictions_adc,predictions_t2w],axis=-1)
concat = tf.keras.layers.concatenate([adc_input,t2w_input],axis=-1)

t2w_input = tf.keras.Input(shape=(predictions_t2w.shape))

dec_adc = Model(concat, models["ADC"].layers[-1](concat))
dec_adc.summary()

#tf.keras.layers.Concatenate(axis=-1)([predictions_adc,predictions_t2w])
#tf.keras.layers.Concatenate(axis=-1[adc_input,t2w_input])

# asd = [models["ADC"].get_layer('decoder_stage0_upsampling'),
#         models["ADC"].get_layer('decoder_stage0_concat')]

# dec_adc = Model(decoder_input, models["ADC"].get_layer('decoder_stage0_upsampling')(decoder_input))



# #dec_adc =  Model(decoder_input, [models["ADC"].layers[-10],models["ADC"].layers[-9]](decoder_input))
# #dec_adc =  Model(decoder_input, models["ADC"].layers[-1](decoder_input))

# dec_adc.summary()

#autoencoder = Sequential([encoder, decoder])

# aut = Model(enc_adc.input, dec_adc(enc_adc.output))
# aut.summary()

# pred_test = tf.repeat(augment_patches(y_val[0]),3,-1)
# predictions = aut.predict(pred_test)
# print(predictions.shape)
#img_pltsave([y_val[0][0], pred_test, predictions])


# pred_test = tf.repeat(augment_patches(y_val[0]),3,-1)
# predictions = models["ADC"].predict(pred_test)
# img_pltsave([y_val[0][0], pred_test, predictions])

# decoder_input = tf.keras.Input(shape=(encoding_dim,))
# decoder = Model(decoder_input, models["ADC"].get_layer('center_block2_relu')(decoder_input))

# dec_adc = Model(models["ADC"].get_layer('center_block2_relu').input, models["ADC"].output)
# dec_t2w = Model(models["t2tsetra"].get_layer('center_block2_relu').input, models["t2tsetra"].output)



#concat = tf.keras.layers.Concatenate([enc_adc, enc_t2w])
#out = tf.keras.layers.Dense(2,activation="sigmoid")(concat)
#model = tf.keras.Model(inputs=[enc_adc.input, enc_t2w.input], outputs=out)
#model.summary()














#%%



# %%
