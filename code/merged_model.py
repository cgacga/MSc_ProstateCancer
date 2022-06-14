import segmentation_models_3D as sm
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.models import Model
from params import modality
import numpy as np
import sys


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
            if l.name.startswith("input"):
                continue               
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
            #print(f"a-{a}, t-{t}, a//t-{max(a,t)//min(a,t)}, a/t-{max(a,t)/min(a,t)}")
            pool_tup.append(max(a,t)//min(a,t))

    mod = {}
    ##UPSIZE##
    if method == "padd":
        #Padding
        size_tuple(max_out)
        mod[min_out] = layers.ZeroPadding3D(padding=(pad_tup),name=f'{min_out}_padding')(models_dict[min_out])
    elif method == "upsample":
        #UpSampling
        size_tuple(max_out)
        mod[min_out] = layers.UpSampling3D(size=(pool_tup),name=f'{min_out}_upsampling')(models_dict[min_out])
    elif method == "transpose":
        #Transpose
        size_tuple(max_out)
        pol2x = [x*2 for x in pool_tup]
        mod[min_out] = layers.Conv3DTranspose(512,pol2x,strides=pool_tup,padding="same",name=f'{min_out}_conv3dtrans')(models_dict[min_out])
        
    ##DOWNSIZE##
    elif method == "crop":
        #Cropping
        size_tuple(min_out)
        mod[max_out] = layers.Cropping3D(cropping=(pad_tup),name=f'{max_out}_cropping')(models_dict[max_out])
    elif method == "reshape":
        #Reshape
        size_tuple(min_out)
        mod[max_out] = layers.Reshape(target_shape=(*models_dict[min_out].shape[1:-1],-1), name=f'{max_out}_reshape')(models_dict[max_out])
    elif method == "maxpool":
        #MaxPooling
        size_tuple(min_out)
        mod[max_out] = layers.MaxPooling3D(pool_size=(pool_tup), strides = pool_tup,name=f'{max_out}_maxpooling')(models_dict[max_out])
    elif method == "avgpool":
        #AveragePooling
        size_tuple(min_out)
        mod[max_out] = layers.AveragePooling3D(pool_size=pool_tup, strides = pool_tup,name=f'{max_out}_averagepooling')(models_dict[max_out])

    return mod






def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    #concat_axis = 4 if backend.image_data_format() == 'channels_last' else 1
    concat_axis = 4

    def firstupsample(input_tensor,skip):
        input_tensor = layers.UpSampling3D(size=2, name=up_name)(input_tensor)
        xskip_dict = {'x':input_tensor,'skip':skip}
        mod = upsizedown(xskip_dict, modality.decoder_method)
        x,skip = [(value if key not in mod.keys() else mod[key]) for key,value in xskip_dict.items()]
        x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])
        return x

    def noupsample(input_tensor,skip):
        xskip_dict = {'x':input_tensor,'skip':skip}
        mod = upsizedown(xskip_dict, modality.decoder_method)
        x,skip = [(value if key not in mod.keys() else mod[key]) for key,value in xskip_dict.items()]
        x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])
        return x

    def wrapper(input_tensor, skip=None):
        if skip is not None:
            x_temp = layers.UpSampling3D(size=2, name=up_name)(input_tensor)
            # print(x_temp.shape)
            # print(x_temp.shape[:-1])
            # print(x_temp.shape[:-2])
            # print(skip.shape)
            # print(skip.shape[:-1])
            # print(skip.shape[:-2])
            if not isinstance(skip, list) and not np.array_equal(x_temp.shape[:-1] , skip.shape[:-1]):
                # xskip_dict = {'x':input_tensor,'skip':skip}
                # mod = upsizedown(xskip_dict, modality.decoder_method)
                
                try:
                    # xskip_dict = {'x':input_tensor,'skip':skip}
                    # mod = upsizedown(xskip_dict, modality.decoder_method)

                    # x,skip = [(value if key not in mod.keys() else mod[key]) for key,value in xskip_dict.items()]
                    # x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])
                    if modality.decode_try_upsample_first:
                        x = firstupsample(input_tensor,skip)
                    else:
                        x = noupsample(input_tensor,skip)
                except ValueError as e:
                    # input_tensor = layers.UpSampling3D(size=2, name=up_name)(input_tensor)
                    # xskip_dict = {'x':input_tensor,'skip':skip}
                    # mod = upsizedown(xskip_dict, modality.decoder_method)
                    # x,skip = [(value if key not in mod.keys() else mod[key]) for key,value in xskip_dict.items()]
                    # x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

                    print("Error: decode_try_upsample_first method not supported")
                    #sys.exit(e)
                    #raise ValueError("Error: decode_try_upsample_first method not supported")

                    
                    
                    # if modality.decode_try_upsample_first:
                    #     x = noupsample(input_tensor,skip)
                    # else:
                    #     x = firstupsample(input_tensor,skip)
                        

            else:
                x = layers.UpSampling3D(size=2, name=up_name)(input_tensor)
                if isinstance(skip, list):
                    x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, *skip])
                else:
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
        if skip is not None:
            x_temp = layers.UpSampling3D(size=2, name=transp_name)(input_tensor)
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
        use_batchnorm=False,
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
        #if False:
            #skip = layers.Concatenate(axis=-1)([skips[i*2],skips[(i*2)+1]]) if i < len(skips)/2 else None
            skip = [skips[i*2],skips[(i*2)+1]] if i < len(skips)/2 else None
            x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)
        else:
            a,t = (a,t) if i != 0 else (x,x)
            adc, t2w = [skips[(i*2)+n] for n in range(2)] if i < len(skips)/2 else (None,None)
            a = decoder_block(decoder_filters[i], stage=f"{i}_{modality.merged_modalities[0]}", use_batchnorm=use_batchnorm)(a, adc)
            t = decoder_block(decoder_filters[i], stage=f"{i}_{modality.merged_modalities[1]}", use_batchnorm=use_batchnorm)(t, t2w)
        #elif mode == makes no sense
        #elif False:
            # if i < len(skips)/2:
        
            #     #skips_dict = {f"adc{i}":skips[i*2],f"t2w{i}":skips[(i*2)+1]}
            #     skips_dict = {f"adc{i}":adc,f"t2w{i}":t2w}
            #     mod = upsizedown(skips_dict, modality.encoder_method, half = False)
            #     t2w,adc = [(value if key not in mod.keys() else mod[key]) for key,value in skips_dict.items()]
            #     skip = layers.Concatenate(axis=-1)([adc,t2w])

            # x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)
            

    
    def xdropout(x, name_suffix=""):
        return layers.SpatialDropout3D(modality.dropout, name=f'{name_suffix}_pyramid_dropout')(x)


    def final_conv(x, name_suffix=""):
        x = layers.Conv3D(
            filters=classes,
            kernel_size=(3, 3, 3),
            padding='same',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name=f'final_conv_{name_suffix}',
        )(x)
        #x = layers.Activation(activation, name=f"{activation}_{name_suffix}")(x)
        #x = layers.Activation(activation, name=f"{name_suffix}")(x)
        return x
    def activation_final(x, name_suffix=""):
        x = layers.Activation(activation, name=f"{name_suffix}")(x)
        return x

    if modality.same_shape:
    #if False:
        if modality.dropout:
            x = xdropout(x,name_suffix="SameShape")
        x = final_conv(x, name_suffix="SameShape")
        if modality.classes > 1:
            model = Model(input_, activation_final(x, name_suffix=modality.activation))
            #x = activation_final(x, name_suffix=modality.activation)
            #model = Model(input_, [layers.Lambda(tf.unstack, arguments=dict(axis=-1))(x)])
        else:
            model = Model(input_, [activation_final(x, name_suffix=f"{name}") for name in modality.merged_modalities])
            #model = Model(input_, x)
    else:
        if modality.dropout:
            a = xdropout(a, name_suffix=f"{modality.merged_modalities[0]}")
            t = xdropout(t, name_suffix=f"{modality.merged_modalities[1]}")
        a = final_conv(a, name_suffix=f"{modality.merged_modalities[0]}")
        a = activation_final(a, name_suffix=f"{modality.merged_modalities[0]}")
        t = final_conv(t, name_suffix=f"{modality.merged_modalities[1]}")
        t = activation_final(t, name_suffix=f"{modality.merged_modalities[1]}")
        model = Model(input_, [a,t])
        
    return model



def get_merged_model():
    encoders = {}
    for i,modality_name in enumerate(modality.merged_modalities):

        #dims = parameters.lst[modality_name]["image_shape"]
        # #dims = modality.image_shape[i]
        # print(modality.image_shape[i])
        # print(isinstance(modality.image_shape[i], tuple))
        # print(modality.reshape_dim[i])
        # print(isinstance(modality.reshape_dim[i], tuple))
        # dims = modality.reshape_dim[i] if modality.reshape_dim[i] != None else modality.image_shape[i]
        dims = modality.image_shape[i] if isinstance(modality.image_shape[i], tuple) else modality.reshape_dim[i]
            
        model = sm.Unet(
            modality.backbone_name, 
            input_shape=(dims[0],dims[1],dims[2],3), 
            encoder_weights=modality.encoder_weights, 
            encoder_freeze = modality.encoder_freeze,
            decoder_use_batchnorm = modality.batchnorm,
            dropout = modality.dropout)
        
        encoder = Model(model.input, model.get_layer(sm.Backbones.get_feature_layers(modality.backbone_name, n=1)[0]).output,name=f'encoder_{modality_name}')     

        rename_all_layers(encoder, modality_name)
        encoders[modality_name] = encoder

    if modality.same_shape:
        concat = layers.Concatenate(axis=-1)([encoders[key].output for key in encoders.keys()])
        concat = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(concat)
    else:
        out_sum = {key: sum(value.output.shape[1:-1]) for key,value in encoders.items()}
        max_out, min_out = [m(out_sum, key = out_sum.get) for m in [max,min]]
        if modality.encoder_method in ["upsample", "transpose", "padd"]:
            enc = max_out
        else: enc = min_out

    if modality.merge_method == "concat":
        merge_method = layers.Concatenate(axis=-1)
    elif modality.merge_method == "add":
        merge_method = layers.Add()
    elif modality.merge_method == "avg":
        merge_method = layers.Average()
    elif modality.merge_method == "max":
        merge_method = layers.Maximum()
    elif modality.merge_method == "multiply":
        merge_method = layers.Multiply()
    
    def firstmaxpool(encoders):
        x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(encoders[enc].output)
        #encoders[enc] = Model(encoders[enc].input, x)
        enc_copy =encoders.copy()
        enc_copy[enc] = Model(enc_copy[enc].input, x)

        mod = upsizedown(enc_copy, modality.encoder_method)

        #concat = layers.Concatenate(axis=-1)([(enc_copy[key].output if key not in mod.keys() else mod[key]) for key in enc_copy.keys()])
        concat = merge_method([(enc_copy[key].output if key not in mod.keys() else mod[key]) for key in enc_copy.keys()])
        
        encoders = enc_copy.copy()
        return concat

    def fistconcat(encoders):
        mod = upsizedown(encoders, modality.encoder_method)

        #concat = layers.Concatenate(axis=-1)([(encoders[key].output if key not in mod.keys() else mod[key]) for key in encoders.keys()])
        concat = merge_method([(encoders[key].output if key not in mod.keys() else mod[key]) for key in encoders.keys()])

        concat = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(concat)
        return concat

    if not modality.same_shape:
        try:
            if modality.encode_try_maxpool_first:
                concat = firstmaxpool(encoders)
            else: concat = fistconcat(encoders)
        except ValueError as e:
            print("Error: encode_try_maxpool_first method not supported")
            #sys.exit(e)
        #raise ValueError("Error: encode_try_maxpool_first method not supported")
        

        # if modality.decode_try_maxpool_first:
        #     concat = fistconcat(encoders)
        # else: concat = firstmaxpool(encoders)
        
        #encoders[enc] = Model(encoders[enc].input, x)




    model = Model(inputs=[enc.input for enc in encoders.values()], outputs=[concat],name="concat_model")

    return build_merged_unet(
                    backbone = model, 
                    decoder_block_type = modality.decoder_block_type, 
                    #decoder_filters=(1024,512,256, 128, 64),
                    #decoder_filters=(512,256, 128, 64, 32), 
                    #decoder_filters=(256, 128, 64, 32, 16), 
                    decoder_filters = modality.decoder_filters,
                    n_upsample_blocks = 5, 
                    classes = modality.classes, 
                    activation = modality.activation, 
                    use_batchnorm = modality.batchnorm, 
                    dropout = modality.dropout
                    )
