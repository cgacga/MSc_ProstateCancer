
### Model building ###
import os, time, json
import tensorflow as tf
from tensorflow import keras
import segmentation_models_3D as sm
sm.set_framework('tf.keras')
from img_display import *
from parameters import modality


class PlotCallback(tf.keras.callbacks.Callback):
    x_val = None
    def __init__(self):
        #self.x_val = PlotCallback.x_val#self.__class__.x_val
        #self.file_writer = tb_callback._val_writer
        #self.x_val = x_val
        self.epoch = 0
        #self.savepath = f"../tb_logs/{os.environ['SLURM_JOB_NAME']}/{modality}/{os.environ['SLURM_JOB_ID']}/{os.environ['SLURM_JOB_NAME']}_{modality}"
        #self.savepath = modality.tensorboard_path

        self.savename = f"{modality.modality_name}_{modality.dtime}"
        
        #modality.model_name
        #f"{modality.job_name}_{modality.modality_name}_{modality.time}"

        self.epoch_modulo = modality.tensorboard_img_epoch

        #logdir = os.path.abspath(self.savepath) #opp ^ (fra params)
        
        #self.path = os.path.abspath(os.path.join(modality.tensorboard_path,f"../"))
        #self.file_writer = tf.summary.create_file_writer(logdir=modality.tensorboard_path)
        #self.file_writer = tf.summary.create_file_writer(logdir=path)
        self.train_writer = tf.summary.create_file_writer(logdir=os.path.join(modality.tensorboard_path,"train"))
        self.val_writer = tf.summary.create_file_writer(logdir=os.path.join(modality.tensorboard_path,"val"))
        #self.file_writer = self.val_writer
        
        
                
    def plot_predictions(self):
        #with modality.strategy.scope():
        
        #pred_sample = tf.repeat(self.model.predict(self.__class__.x_val),3,-1)
        
        pred_sample = np.zeros_like(self.__class__.x_val)
        for i,xval in enumerate(self.__class__.x_val):
            #pred_sample = tf.concat([pred_sample,tf.repeat(self.model.predict(xval),3,-1)],0)
            pred_sample[i] = self.model.predict(tf.repeat(tf.expand_dims(xval,0),3,-1))
        
        img_stack = np.stack([self.__class__.x_val,pred_sample],0+1).reshape(*(-(0==j) or s for j,s in enumerate(pred_sample.shape)))

        image = img_pltsave([img for img in img_stack],None,True)
        #with modality.strategy.scope():
        with self.val_writer.as_default():
            tf.summary.image(modality.job_name+f"_img/{self.savename}", image, step=self.epoch, description=modality.mrkdown())
            self.val_writer.flush()

    def json_summary(self,logs):
        
        modality.training_logs = logs
        #with modality.strategy.scope():
        with self.val_writer.as_default():
            tf.summary.text(modality.job_name+f"_txt/{self.savename}", modality.mrkdown(), step=self.epoch, description=modality.model_name)
            self.val_writer.flush()
        
        
    def on_epoch_end(self, epoch, logs={}):           
        self.epoch = epoch+1     
         
        train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        val_logs = {k: v for k, v in logs.items() if k.startswith('val_')}

        # for key in logs:
        #     with self.file_writer.as_default():
        #         tf.summary.scalar(f"{modality.model_name}/{key}", logs[key], step=self.epoch, description=f"{key}_{modality.model_name}") 
        #         self.file_writer.flush()   

        if train_logs:
            with self.train_writer.as_default():
                for key, value in train_logs.items():
                    tf.summary.scalar(f"{modality.modality_name}/{key}",value,self.epoch)#, description=f"{key}_{modality.model_name}")
                    #summary_ops_v2.scalar('epoch_' + name, value, step=epoch)
                self.train_writer.flush()
        if val_logs:
            with self.val_writer.as_default():
                for key, value in val_logs.items():
                    key = key[4:]  # Remove 'val_' prefix.
                    tf.summary.scalar(f"{modality.modality_name}/{key}",value,self.epoch)#, description=f"{key}_{modality.model_name}") 
                    #summary_ops_v2.scalar('epoch_' + name, value, step=epoch)
                self.val_writer.flush()
        
        if self.epoch%self.epoch_modulo==0:
            self.plot_predictions()
            

    def on_train_end(self, logs={}):

        if self.epoch%self.epoch_modulo!=0:
            self.plot_predictions()
        
        self.json_summary(logs)


def train_model(model, trainDS, valDS):
    """
    Function to train the model.
    
    :param model: The model to train
    :param trainDS: Training data
    :param valDS: Validation data
    :return: The trained model.
    """
    #print("\n"+f"{model.name} - training started".center(50, '.'))
    print("\n"+f"{modality.model_name} - training started".center(50, '.'))
    #print("\n"+f"{settings.current.model_name} - training started".center(50, '.'))


    #savepath = f"../models/{os.environ['SLURM_JOB_NAME']}/{modality}/"
    
    #checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(savepath,f"checkpoint_cb/{model.name}-checkpoint.h5"), save_best_only=True)

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


    # tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #     #log_dir=f"../tb_logs/{os.environ['SLURM_JOB_NAME']}/{modality}",
    #     #log_dir=f"../tb_logs/{os.environ['SLURM_JOB_NAME']}/{modality}/{os.environ['SLURM_JOB_ID']}",
    #     log_dir=modality.tensorboard_path,
    #     histogram_freq=1,
    #     write_graph=True,
    #     write_images=False,
    #     write_steps_per_second=False,
    #     update_freq='epoch',
    #     profile_batch=0,
    #     embeddings_freq=0,
    #     embeddings_metadata=None)



        
                
                    


    
    
    #x_val = list(valDS.map(lambda x,y: (x[0:modality.tensorboard_num_predictimages])))[0]
    #x_val = list(valDS.as_numpy_iterator())[0][0][0:modality.tensorboard_num_predictimages]
    pltcallback = PlotCallback()

    model.fit(
        # x=train_data,
        trainDS,
        validation_data = valDS,
        epochs = modality.epochs,
        #batch_size=batch_size,
        verbose=2,
        
        # callbacks=[checkpoint_cb, early_stopping_cb],
        #callbacks=early_stopping_cb,
        #steps_per_epoch =  modality.steps_pr_epoch,
        #validation_steps =  modality.validation_steps,
        callbacks = [pltcallback],
        #callbacks = [tensorboard_callback,pltcallback],
    )

    # print("\n"+f"{model.name} - Train accuracy:\t {round (model.history['acc'][0], 4)}".center(50, '.'))

    #try:   
    #    model.summary()
    #except:
    #    print("\n"+f"{model.name} - Model summary failed".center(50, '.'))

    return model

def get_unet_model(input_shape, modality_name="autoencoder"):

    # shape = (width, height, depth, 1)
    print("\n"+f"{modality.model_name} - compile unet model".center(50, '.'))
    #keras.backend.clear_session()

    #TODO: sbatch variable for unet backbone
    #autoencoder = sm.Unet("vgg16", input_shape=(dim[0], dim[1], dim[2],3), encoder_weights="imagenet")
    autoencoder = sm.Unet(
        backbone_name = modality.backbone_name, #"vgg16",
        input_shape = (None,None,None,3),#modality.input_shape,
        encoder_weights = modality.encoder_weights,# "imagenet",
        classes = modality.classes,# 1,
        encoder_freeze = modality.encoder_freeze,# False,
        decoder_block_type = modality.decoder_block_type,# "transpose",
        activation = modality.activation)# "sigmoid")
    #https://github.com/ZFTurbo/segmentation_models_3D/blob/master/segmentation_models_3D/models/unet.py#L166

    #TODO: sbatch variable for learning_rate
    #opt = tf.keras.optimizers.Adam()
    #learning_rate = opt.lr.numpy()*len(tf.config.list_physical_devices('GPU'))
    #opt.lr.assign(learning_rate)

    #autoencoder._name = f"{modality_name}_{os.environ['SLURM_JOB_NAME']}"
    autoencoder._name = modality.model_name#f"{modality_name}_{modality.job_name}"

    #TODO: check loss function?
    #TODO: add MSE and MAE for each epoch and save log
    #TODO: add loss and val loss function for each epoch

    #reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE #SUM #NONE 
    #https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanAbsoluteError
    #mse = tf.keras.losses.MeanSquaredError()#reduction)
    #mae = tf.keras.losses.MeanAbsoluteError()#reduction)
    

    autoencoder.compile(modality.optimizer, loss=modality.loss, metrics=modality.metrics)


    print(autoencoder.summary())

    

    return autoencoder



def model_building(trainDS, valDS ):
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
    # gpus = tf.config.list_physical_devices('GPU')
    # strategy = tf.distribute.MirroredStrategy()
    # if len(gpus)>1:

    #     print("\nprint two gpus\n")
        
    #     with strategy.scope():
    #         #model = get_unet_model(shape, f"{os.environ['SLURM_JOB_NAME']}")
    #         model = get_unet_model(modality.shape,modality.modality_name)
            

    # else: 
    #     # model = get_model(shape, f"{os.environ['SLURM_JOB_NAME']}")
    #     # shape=x_data.shape
    #     #model = get_unet_model(shape, f"{os.environ['SLURM_JOB_NAME']}")
    #     model = get_unet_model(modality.shape,modality.modality_name)
        #model = get_unet_model(shape, f"asd")

    #with modality.strategy.scope():
    #        model = get_unet_model(modality.image_shape,modality.modality_name)
    model = get_unet_model(modality.image_shape,modality.modality_name)

    #TODO: load selected weights from sbatch variable if exists
    
    #model = train_model(model,x_data[idx], y_data[idx])
    #savepath = f"../models/{os.environ['SLURM_JOB_NAME']}/{modality}"
    #savepath = f"../models/{os.environ['SLURM_JOB_NAME']}/{modality}/{os.environ['SLURM_JOB_ID']}/"
    save_path = os.path.join(modality.model_path,modality.model_name)
    #with modality.strategy.scope():
    if os.path.isdir(modality.model_path): 
        #if len(gpus)>1:
        #    with strategy.scope():
                # model.load_weights(os.path.join(savepath,"weights"))    
                #model.load_weights(os.path.join(savepath,modality)) #savepath)
        #        model.load_weights(save_path)
        #        print("Loaded Weights")
        #else: #model.load_weights(os.path.join(savepath,modality)) 
        #    model.load_weights(save_path)       
        #with modality.strategy.scope:
        model.load_weights(save_path)
        print("Loaded Weights")
    else: 
        model = train_model(model, trainDS, valDS)
        model.save(save_path)   
        #model.save_weights(os.path.join(savepath,modality))

    #TODO: save images from recreation, with noisy and clean included (use more than 5 images in the plot)
        
    # model.save(f"../models/{model.name}/{os.environ['SLURM_JOB_NAME']}/{os.environ['SLURM_JOB_ID']}-{os.environ['SLURM_JOB_NAME']}")

    print("\n"+f" Model building finished {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

    return model

