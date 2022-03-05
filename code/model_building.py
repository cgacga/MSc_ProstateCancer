
### Model building ###
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import pandas as pd
import time

def get_model(dim, name="autoencoder"):
    """
    It creates a model that accepts a 3D input of shape (width, height, depth, 1) and returns a 3D
    output of the same shape.
    
    :param dim: The shape of the input data
    :param name: The Model's name
    :return: The autoencoder model
    """
    # shape = (width, height, depth, 1)
    print("\n"+f"{name} - compile model".center(50, '.'))
    #keras.backend.clear_session()
    
    inputs = keras.Input(shape=(dim[0], dim[1], dim[2], 1))

    # Encoder
    x = layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling3D((2, 2, 2), padding="same")(x)
    x = layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(x)
    encoded = layers.MaxPooling3D((2, 2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv3DTranspose(32, (3, 3, 3), strides=2, activation="relu", padding="same")(encoded)
    x = layers.Conv3DTranspose(32, (3, 3, 3), strides=2, activation="relu", padding="same")(x)
    decoded = layers.Conv3D(1, (3, 3, 3), activation="sigmoid", padding="same")(x)

    # Autoencoder
    autoencoder = Model(inputs, decoded, name=name)
    opt = tf.keras.optimizers.Adam()
    learning_rate = opt.lr.numpy()*len(tf.config.list_physical_devices('GPU'))
    opt.lr.assign(learning_rate)
    # print(f"Learning rate = {opt.lr.numpy()}")
    autoencoder.compile(opt, loss="binary_crossentropy")

    print(autoencoder.summary())

    return autoencoder


def train_model(model, path, train_data, val_data):
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
    checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(path,f"{model.name}-checkpoint.h5"), save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)

    batch_size = 16
    #32 -> 34147MiB / 40536MiB (error)
    #16 ->  34147MiB / 40536MiB (no error? check logs)
    #8  -> 17819MiB / 40536MiB
    print(f"Batch size = {batch_size}")

    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)
    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options)
    
    model.fit(
        x=train_data,
        epochs=50,
        batch_size=batch_size,
        verbose=2,
        validation_data=val_data,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )

    # print("\n"+f"{model.name} - Train accuracy:\t {round (model.history['acc'][0], 4)}".center(50, '.'))
    

    return model



def model_building(shape, savepath, x_data, y_data ):
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

    

    # shape,idx = patients_df[["dim","tag_idx"]][patients_df.tag.str.contains(modality, case=False)].values[0]
    if len(tf.config.list_physical_devices('GPU'))>1:
        with tf.distribute.MirroredStrategy().scope():
            model = get_model(shape, f"{os.environ['SLURM_JOB_NAME']}")
    else: model = get_model(shape, f"{os.environ['SLURM_JOB_NAME']}")
    
    #model = train_model(model,x_data[idx], y_data[idx])
    if os.path.exists(savepath): 
        model.load_weights(savepath)
    else: 
        model = train_model(model, savepath, x_data, y_data)
        model.save(savepath)
    # model.save(f"../models/{model.name}/{os.environ['SLURM_JOB_NAME']}/{os.environ['SLURM_JOB_ID']}-{os.environ['SLURM_JOB_NAME']}")

    print("\n"+f" Model building finished {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

    return model

# def model_evaulation(model, )