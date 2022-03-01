
### Model building ###
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

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
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    print(autoencoder.summary())
    return autoencoder


def train_model(model, x_data, y_data):
    """
    Function to train the model.
    
    :param model: The model to train
    :param x_data: The training data
    :param y_data: The labels for the training data
    :return: The trained model.
    """
    print("\n"+f"{model.name} - training started".center(50, '.'))

    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(f"{model.name} - 3d_image_classification.h5", save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

    batch_size = 2
    print(f"Batch size = {batch_size}")
    
    model.fit(
        x=x_data,
        y=y_data,
        epochs=50,
        batch_size=32,
        verbose=2,
        validation_data=(x_data, y_data),
        callbacks=[checkpoint_cb, early_stopping_cb],
    )

    print("\n"+f"{model.name} - Train accuracy:\t {round (model.history['acc'][0], 4)}".center(50, '.'))
    

    return model


