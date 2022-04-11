
### Model building ###
import os
from xml.dom import VALIDATION_ERR
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
# os.environ["KERAS_BACKEND"] = "tensorflow"
import segmentation_models_3D as sm
sm.set_framework('tf.keras')
import pandas as pd
import time

# def get_model(dim, name="autoencoder"):
#     #TODO: remove this
#     """
#     It creates a model that accepts a 3D input of shape (width, height, depth, 1) and returns a 3D
#     output of the same shape.
    
#     :param dim: The shape of the input data
#     :param name: The Model's name
#     :return: The autoencoder model
#     """
#     # shape = (width, height, depth, 1)
#     print("\n"+f"{name} - compile model".center(50, '.'))
#     #keras.backend.clear_session()
    
#     inputs = keras.Input(shape=(dim[0], dim[1], dim[2], 1))

#     # Encoder
#     x = layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(inputs)
#     x = layers.MaxPooling3D((2, 2, 2), padding="same")(x)
#     x = layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(x)
#     encoded = layers.MaxPooling3D((2, 2, 2), padding="same")(x)

#     # Decoder
#     x = layers.Conv3DTranspose(32, (3, 3, 3), strides=2, activation="relu", padding="same")(encoded)
#     x = layers.Conv3DTranspose(32, (3, 3, 3), strides=2, activation="relu", padding="same")(x)
#     decoded = layers.Conv3D(1, (3, 3, 3), activation="sigmoid", padding="same")(x)

#     # Autoencoder
#     autoencoder = Model(inputs, decoded, name=name)
#     opt = tf.keras.optimizers.Adam()
#     learning_rate = opt.lr.numpy()*len(tf.config.list_physical_devices('GPU'))
#     opt.lr.assign(learning_rate)
#     # print(f"Learning rate = {opt.lr.numpy()}")
#     autoencoder.compile(opt, loss="binary_crossentropy")

#     print(autoencoder.summary())

#     return autoencoder


# def train_model(model, path, train_data, val_data):
def train_model(model, path, x_train_noisy,x_train, x_test_noisy, x_test):
    """
    Function to train the model.
    
    :param model: The model to train
    :param x_data: The training data
    :param y_data: The labels for the training data
    :return: The trained model.
    """
    print("\n"+f"{model.name} - training started".center(50, '.'))

    # Define callbacks.
    # checkpoint_cb = keras.callbacks.ModelCheckpoint(f"../models/{model.name}/{os.environ['SLURM_JOB_NAME']}/{model.name}-checkpoint.h5", save_best_only=True)

    # tf.keras.utils.plot_model(model,show_shapes=True,to_file=os.path.join(path,"model.png"))
    #intall pydot and graphviz
    
    checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(path,f"checkpoint_cb/{model.name}-checkpoint.h5"), save_best_only=True)

    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)

    batch_size = 2
    #32 -> 34147MiB / 40536MiB (2 gpu -> 16 each)
    #16 ->  34147MiB / 40536MiB 
    #8  -> 17819MiB / 40536MiB
    print(f"Batch size = {batch_size}")

    # train_data = train_data.batch(batch_size)
    # val_data = val_data.batch(batch_size)
    # # Disable AutoShard.
    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    # train_data = train_data.with_options(options)
    # val_data = val_data.with_options(options)
    
    print(f"x_train_noisy shape = {x_train_noisy.shape}")

    # from tensorflow.python.ops.numpy_ops import np_config
    # np_config.enable_numpy_behavior()

    #TODO: save weights pr epoch 
    #TODO: save metric with epochs
    #TODO: online loading of data with or without online augmentation


    model.fit(
        # x=train_data,
        x=x_train_noisy,
        y=x_train,
        epochs=500,
        batch_size=batch_size,
        verbose=2,
        # validation_data=val_data,
        validation_data=(x_test_noisy,x_test),
        # callbacks=[checkpoint_cb, early_stopping_cb],
        callbacks=early_stopping_cb,
    )

    # print("\n"+f"{model.name} - Train accuracy:\t {round (model.history['acc'][0], 4)}".center(50, '.'))
    

    return model

def get_unet_model(dim, name="autoencoder"):

    # shape = (width, height, depth, 1)
    print("\n"+f"{name} - compile unet model".center(50, '.'))
    #keras.backend.clear_session()

    #TODO: sbatch variable for unet backbone
    autoencoder = sm.Unet("vgg16", input_shape=(dim[0], dim[1], dim[2],3), encoder_weights="imagenet")

    #TODO: sbatch variable for learning_rate
    opt = tf.keras.optimizers.Adam()
    learning_rate = opt.lr.numpy()*len(tf.config.list_physical_devices('GPU'))
    opt.lr.assign(learning_rate)

    #TODO: check loss function?
    #TODO: add MSE and MAE for each epoch and save log
    #TODO: add loss and val loss function for each epoch
    autoencoder.compile(opt, loss="binary_crossentropy")

    print(autoencoder.summary())

    print("\n done with compiling")

    return autoencoder


def model_building(shape, savepath, x_data, y_data, x_val,y_val ):
    """
    Given a dataframe of patients, a modality (e.g. "ct"), and the corresponding x and y data, 
    this function returns a trained model for the modality
    
    :param patients_df: The dataframe containing the patient data
    :type patients_df: pd.DataFrame
    :param modality: The modality you want to train the model on
    :type modality: str
    :param x_data: The x-data is the input data for the model. In this case, it's the MRI images
    :param y_data: The target data
    :return: The model
    """
    print(f" Model building started".center(50, '_'))
    start_time = time.time()

    
    #TODO: concatinate training and validation sets to one 30% set
    
    # shape,idx = patients_df[["dim","tag_idx"]][patients_df.tag.str.contains(modality, case=False)].values[0]
    gpus = tf.config.list_physical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy()
    if len(gpus)>1:

        print("\nprint two gpus\n")
        
        with strategy.scope():
            # model = get_model(shape, f"{os.environ['SLURM_JOB_NAME']}")
            model = get_unet_model(shape, f"{os.environ['SLURM_JOB_NAME']}")

    else: 
        # model = get_model(shape, f"{os.environ['SLURM_JOB_NAME']}")
        # shape=x_data.shape
        model = get_unet_model(shape, f"{os.environ['SLURM_JOB_NAME']}")
        


    #TODO: load selected weights from sbatch variable if exists
    
    #model = train_model(model,x_data[idx], y_data[idx])
    if os.path.isdir(savepath): 
        if len(gpus)>1:
            with strategy.scope():
                # model.load_weights(os.path.join(savepath,"weights"))    
                model.load_weights(savepath)
                print("Loaded Weights")
        else: model.load_weights(os.path.join(savepath,"weights"))        
    else: 
        model = train_model(model, savepath, x_data, y_data, x_val, y_val)
        # model.save(savepath)
        model.save_weights(os.path.join(savepath,"weights"))

    #TODO: save images from recreation, with noisy and clean included (use more than 5 images in the plot)
        
    # model.save(f"../models/{model.name}/{os.environ['SLURM_JOB_NAME']}/{os.environ['SLURM_JOB_ID']}-{os.environ['SLURM_JOB_NAME']}")

    print("\n"+f" Model building finished {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

    return model

