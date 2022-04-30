
### Model building ###
import os, time, json
import tensorflow as tf
from tensorflow import keras
import segmentation_models_3D as sm
sm.set_framework('tf.keras')
from img_display import *

def train_model(model, modality, trainDS, valDS):
    """
    Function to train the model.
    
    :param model: The model to train
    :param trainDS: Training data
    :param valDS: Validation data
    :return: The trained model.
    """
    print("\n"+f"{model.name} - training started".center(50, '.'))

    # Define callbacks.
    # checkpoint_cb = keras.callbacks.ModelCheckpoint(f"../models/{model.name}/{os.environ['SLURM_JOB_NAME']}/{model.name}-checkpoint.h5", save_best_only=True)

    # tf.keras.utils.plot_model(model,show_shapes=True,to_file=os.path.join(path,"model.png"))
    #intall pydot and graphviz
    savepath = f"../models/{os.environ['SLURM_JOB_NAME']}/{modality}/"
    
    checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(savepath,f"checkpoint_cb/{model.name}-checkpoint.h5"), save_best_only=True)

    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)

    #batch_size = 2
    

    # train_data = train_data.batch(batch_size)
    # val_data = val_data.batch(batch_size)
    # # Disable AutoShard.
    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    # train_data = train_data.with_options(options)
    # val_data = val_data.with_options(options)
    

    # from tensorflow.python.ops.numpy_ops import np_config
    # np_config.enable_numpy_behavior()

    #TODO: save weights pr epoch 
    #TODO: save metric with epochs
    #TODO: online loading of data with or without online augmentation


    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        #log_dir=f"../tb_logs/{os.environ['SLURM_JOB_NAME']}/{modality}",
        log_dir=f"../tb_logs/{os.environ['SLURM_JOB_NAME']}/{modality}/{os.environ['SLURM_JOB_ID']}",
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        write_steps_per_second=False,
        update_freq='epoch',
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None)



    class PlotCallback(tf.keras.callbacks.Callback):
        def __init__(self, x_val, params):
            self.x_val = x_val
            self.epoch = 0
            self.savepath = f"../tb_logs/{os.environ['SLURM_JOB_NAME']}/{modality}/{os.environ['SLURM_JOB_ID']}/{os.environ['SLURM_JOB_NAME']}_{modality}"
            self.params = params

            self.epoch_modulo = 10

            logdir = os.path.abspath(self.savepath) #opp ^ (fra params)
            self.file_writer = tf.summary.create_file_writer(logdir)
                    
        def plot_predictions(self):
                
                pred_sample = tf.repeat(self.model.predict(self.x_val),3,-1)
                img_stack = np.stack([self.x_val,pred_sample],0+1).reshape(*(-(0==j) or s for j,s in enumerate(pred_sample.shape)))

                image = img_pltsave([img for img in img_stack],None,True)

                #logdir = f"/bhome/cga2022/jobs/tb_logs/{savepath}"
                with self.file_writer.as_default():
                    tf.summary.image(f"name_{modality}", image, step=self.epoch)

        def json_summary(self,logs):
            
            params = self.params
            params["logs"] = logs
            json_dict = json.dumps(params, indent=2)
            summary = "".join("\t" + line for line in json_dict.splitlines(True))
            
            with self.file_writer.as_default():
                tf.summary.text("modell_navn", summary, step=self.epoch)

            
        def on_epoch_end(self, epoch, logs=None):           
            self.epoch = epoch+1                
            
            if self.epoch%self.epoch_modulo==0:
                self.plot_predictions(self)
                

        def on_train_end(self, logs={}):

            if not self.epoch%self.epoch_modulo==0:
                self.plot_predictions(self)

            self.json_summary(self,logs)
            
                


    
    #x_val,y_val = list(valDS.map(lambda x,y: (x[0:5], tf.repeat(y[0:5],3,-1))))[0]
    x_val = list(valDS.map(lambda x,y: (x[0:5])))[0]
    pltcallback = PlotCallback(x_val)

    model.fit(
        # x=train_data,
        trainDS,
        validation_data=valDS,
        epochs=100,
        #batch_size=batch_size,
        verbose=2,
        
        # callbacks=[checkpoint_cb, early_stopping_cb],
        #callbacks=early_stopping_cb,
        callbacks=[tensorboard_callback,early_stopping_cb,pltcallback],
    )

    # print("\n"+f"{model.name} - Train accuracy:\t {round (model.history['acc'][0], 4)}".center(50, '.'))

    #try:   
    #    model.summary()
    #except:
    #    print("\n"+f"{model.name} - Model summary failed".center(50, '.'))

    return model

def get_unet_model(dim, modality="autoencoder"):

    # shape = (width, height, depth, 1)
    print("\n"+f"{modality} - compile unet model".center(50, '.'))
    #keras.backend.clear_session()

    #TODO: sbatch variable for unet backbone
    #autoencoder = sm.Unet("vgg16", input_shape=(dim[0], dim[1], dim[2],3), encoder_weights="imagenet")
    autoencoder = sm.Unet(
        backbone_name = "vgg16",
        input_shape=(None,None,None,3),
        encoder_weights="imagenet",
        classes = 1,
        encoder_freeze=False,
        decoder_block_type="transpose",
        activation="sigmoid")
    #https://github.com/ZFTurbo/segmentation_models_3D/blob/master/segmentation_models_3D/models/unet.py#L166

    #TODO: sbatch variable for learning_rate
    opt = tf.keras.optimizers.Adam()
    learning_rate = opt.lr.numpy()*len(tf.config.list_physical_devices('GPU'))
    opt.lr.assign(learning_rate)

    autoencoder._name = f"{modality}_{os.environ['SLURM_JOB_NAME']}"

    #TODO: check loss function?
    #TODO: add MSE and MAE for each epoch and save log
    #TODO: add loss and val loss function for each epoch

    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE #SUM #NONE 
    #https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanAbsoluteError
    mse = tf.keras.losses.MeanSquaredError()#reduction)
    mae = tf.keras.losses.MeanAbsoluteError()#reduction)
    

    autoencoder.compile(opt, loss=mse, metrics=["mse","mae"])


    print(autoencoder.summary())

    print("\n done with compiling")

    return autoencoder



def model_building(shape, modality, trainDS, valDS ):
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
    gpus = tf.config.list_physical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy()
    if len(gpus)>1:

        print("\nprint two gpus\n")
        
        with strategy.scope():
            #model = get_unet_model(shape, f"{os.environ['SLURM_JOB_NAME']}")
            model = get_unet_model(shape, modality)
            

    else: 
        # model = get_model(shape, f"{os.environ['SLURM_JOB_NAME']}")
        # shape=x_data.shape
        #model = get_unet_model(shape, f"{os.environ['SLURM_JOB_NAME']}")
        model = get_unet_model(shape, modality)
        #model = get_unet_model(shape, f"asd")

    #TODO: load selected weights from sbatch variable if exists
    
    #model = train_model(model,x_data[idx], y_data[idx])
    #savepath = f"../models/{os.environ['SLURM_JOB_NAME']}/{modality}"
    savepath = f"../models/{os.environ['SLURM_JOB_NAME']}/{modality}/{os.environ['SLURM_JOB_ID']}/"
    

    if os.path.isdir(savepath): 
        if len(gpus)>1:
            with strategy.scope():
                # model.load_weights(os.path.join(savepath,"weights"))    
                model.load_weights(os.path.join(savepath,modality)) #savepath)
                print("Loaded Weights")
        else: model.load_weights(os.path.join(savepath,modality))        
    else: 
        model = train_model(model, modality, trainDS, valDS)
        # model.save(savepath)
        model.save_weights(os.path.join(savepath,modality))

    #TODO: save images from recreation, with noisy and clean included (use more than 5 images in the plot)
        
    # model.save(f"../models/{model.name}/{os.environ['SLURM_JOB_NAME']}/{os.environ['SLURM_JOB_ID']}-{os.environ['SLURM_JOB_NAME']}")

    print("\n"+f" Model building finished {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

    return model

