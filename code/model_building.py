
### Model building ###
import os, time, json, statistics
from tabnanny import verbose
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import segmentation_models_3D as sm
sm.set_framework('tf.keras')
from img_display import *
from params import modality
from merged_model import get_merged_model
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.utils import resample
from sklearn.metrics import classification_report
from tensorboard.plugins.custom_scalar import summary as cs_summary
from tensorboard.plugins.custom_scalar import layout_pb2
import seaborn as sns

class PlotCallback(tf.keras.callbacks.Callback):
    x_val = None

    def __init__(self):
        self.modeltype = modality.modeltype
        #self.x_val = PlotCallback.x_val#self.__class__.x_val
        #self.file_writer = tb_callback._val_writer
        #self.x_val = x_val
        self.epoch = 0
        #self.savepath = f"../tb_logs/{os.environ['SLURM_JOB_NAME']}/{modality}/{os.environ['SLURM_JOB_ID']}/{os.environ['SLURM_JOB_NAME']}_{modality}"
        #self.savepath = modality.tensorboard_path

        #self.savename = f"{modality.modality_name}_{modality.dtime}"
        self.savename = f"{modality.modality_name}_{self.modeltype}"
        
        #modality.model_name
        #f"{modality.job_name}_{modality.modality_name}_{modality.time}"

        self.epoch_modulo = modality.tensorboard_img_epoch
        self.nimages = modality.tensorboard_num_predictimages
        self.job_name = modality.job_name
        self.description = modality.mrkdown()
        self.model_name = modality.model_name
        self.modality_name = modality.modality_name

        #logdir = os.path.abspath(self.savepath) #opp ^ (fra params)
        
        #self.path = os.path.abspath(os.path.join(modality.tensorboard_path,f"../"))
        #self.file_writer = tf.summary.create_file_writer(logdir=modality.tensorboard_path)
        #self.file_writer = tf.summary.create_file_writer(logdir=path)
        self.train_writer = tf.summary.create_file_writer(logdir=os.path.join(modality.tensorboard_path,f"{self.modeltype}_train"))
        self.val_writer = tf.summary.create_file_writer(logdir=os.path.join(modality.tensorboard_path,f"{self.modeltype}_val"))
        #self.file_writer = self.val_writer



        # if isinstance(self.__class__.x_val, list):
        #     self.plotting = self.plot_merged_predictions
        # else:
        #     self.plotting = self.plot_predictions
        
    def plot_predictions(self):
        
        #pred_sample = tf.repeat(self.model.predict(self.__class__.x_val),3,-1)
        
        pred_sample = np.zeros_like(self.__class__.x_val)
        for i,xval in enumerate(self.__class__.x_val):
            #pred_sample = tf.concat([pred_sample,tf.repeat(self.model.predict(xval),3,-1)],0)
            pred_sample[i] = self.model.predict(tf.repeat(tf.expand_dims(xval,0),3,-1))
            #pred_sample[i] = tf.repeat(self.model.predict(tf.expand_dims(xval,0)),3,-1)
        
        img_stack = np.stack([self.__class__.x_val,pred_sample],0+1).reshape(*(-(0==j) or s for j,s in enumerate(pred_sample.shape)))

        image = img_pltsave([img for img in img_stack],None,True)
        
        with self.val_writer.as_default():
            tf.summary.image(self.job_name+f"_img/{self.savename}", image, step=self.epoch, description=self.description)
            self.val_writer.flush()

    def plot_merged_predictions(self):

        pred_sample = [np.zeros_like(self.__class__.x_val[i]) for i in range(len(self.__class__.x_val))]

        for i in range(self.nimages):
            img = self.model.predict([tf.repeat(tf.expand_dims(xval[i],0),3,-1) for xval in self.__class__.x_val])
            for j in range(len(img)):
                pred_sample[j][i] = img[j]

        for i, name in enumerate(modality.merged_modalities):        
            
            img_stack = np.stack([self.__class__.x_val[i],pred_sample[i]],0+1).reshape(*(-(0==j) or s for j,s in enumerate(pred_sample[i].shape)))
            
            image = img_pltsave([img for img in img_stack],None,True)
            with self.val_writer.as_default():
                tf.summary.image(self.job_name+f"_img/{name}_{self.savename}", image, step=self.epoch, description=self.description)
                self.val_writer.flush()


    def json_summary(self,logs):
        
        modality.training_logs = logs
        #with modality.strategy.scope():
        with self.val_writer.as_default():
            tf.summary.text(self.job_name+f"_txt/{self.savename}", self.description, step=self.epoch, description=self.model_name)
            self.val_writer.flush()

    def summary(self,logs):
        train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        val_logs = {k: v for k, v in logs.items() if k.startswith('val_')}

        if train_logs:
            with self.train_writer.as_default():
                for key, value in train_logs.items():
                    tf.summary.scalar(f"{self.modality_name}/{self.modeltype}_{key}",value,self.epoch)#
                    
                self.train_writer.flush()
        if val_logs:
            with self.val_writer.as_default():
                for key, value in val_logs.items():

                    tf.summary.scalar(f"{self.modality_name}/{self.modeltype}_{key[4:]}",value,self.epoch)#
                    
                self.val_writer.flush()

    def summary_merged(self,logs):
        train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        val_logs = {k: v for k, v in logs.items() if k.startswith('val_')}

        # for key in logs:
        #     with self.file_writer.as_default():
        #         tf.summary.scalar(f"{modality.model_name}/{key}", logs[key], step=self.epoch, description=f"{key}_{modality.model_name}") 
        #         self.file_writer.flush()   

        if train_logs:
            with self.train_writer.as_default():
                for key, value in train_logs.items():
                    for name in modality.merged_modalities:
                        if key.startswith(name):
                            tf.summary.scalar(f"{name}/{self.modeltype}_{key[len(name)+1:]}",value,self.epoch)
                    if key.startswith("loss"):
                        tf.summary.scalar(f"{self.modality_name}/{self.modeltype}_global_{key}",value,self.epoch)
                    
                    #if key != "learning_rate":
                    #    tf.summary.scalar(f"{modality.modality_name}/train_{key}",value,self.epoch)#
                    #summary_ops_v2.scalar('epoch_' + name, value, step=epoch)
                self.train_writer.flush()
        if val_logs:
            with self.val_writer.as_default():
                for key, value in val_logs.items():
                    key = key[4:]
                    for name in modality.merged_modalities:
                        if key.startswith(name):
                            tf.summary.scalar(f"{name}/{self.modeltype}_{key[len(name)+1:]}",value,self.epoch)
                    if key.startswith("loss"):
                        tf.summary.scalar(f"{self.modality_name}/{self.modeltype}_global_{key}",value,self.epoch)#
                    #key = key[4:]  # Remove 'val_' prefix.
                    #tf.summary.scalar(f"{modality.modality_name}/{key}",value,self.epoch)#
                    #summary_ops_v2.scalar('epoch_' + name, value, step=epoch)
                self.val_writer.flush()

        
        
    def on_epoch_end(self, epoch, logs={}):           
        self.epoch = epoch+1     

        #lr = tf.keras.backend.eval(self.model.optimizer.lr)
        #logs.update({'learning_rate': lr})
        
        
        # train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        # val_logs = {k: v for k, v in logs.items() if k.startswith('val_')}

        # # for key in logs:
        # #     with self.file_writer.as_default():
        # #         tf.summary.scalar(f"{modality.model_name}/{key}", logs[key], step=self.epoch, description=f"{key}_{modality.model_name}") 
        # #         self.file_writer.flush()   

        # if train_logs:
        #     with self.train_writer.as_default():
        #         for key, value in train_logs.items():
        #             tf.summary.scalar(f"{modality.modality_name}/{key}",value,self.epoch)#
        #             #if key != "learning_rate":
        #             #    tf.summary.scalar(f"{modality.modality_name}/train_{key}",value,self.epoch)#
        #             #summary_ops_v2.scalar('epoch_' + name, value, step=epoch)
        #         self.train_writer.flush()
        # if val_logs:
        #     with self.val_writer.as_default():
        #         for key, value in val_logs.items():
        #             tf.summary.scalar(f"{modality.modality_name}/{key[4:]}",value,self.epoch)#
        #             #key = key[4:]  # Remove 'val_' prefix.
        #             #tf.summary.scalar(f"{modality.modality_name}/{key}",value,self.epoch)#
        #             #summary_ops_v2.scalar('epoch_' + name, value, step=epoch)
        #         self.val_writer.flush()
        
        #if self.epoch%self.epoch_modulo==0:
            # if isinstance(self.__class__.x_val, list):
            #     self.plot_merged_predictions()
            # else:
            #     self.plot_predictions()
        
        
        if isinstance(self.__class__.x_val, list):
            self.summary_merged(logs)
            if self.modeltype == "autoencoder":
                if self.epoch%self.epoch_modulo==0:
                    self.plot_merged_predictions()
        else:
            self.summary(logs)
            if self.modeltype == "autoencoder":
                if self.epoch%self.epoch_modulo==0:
                    self.plot_predictions()
        
            

    def on_train_end(self, logs={}):

        if self.modeltype == "autoencoder":
            if self.epoch%self.epoch_modulo!=0:
                if isinstance(self.__class__.x_val, list):
                    self.plot_merged_predictions()
                else:
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

def get_unet_model(modality_name="autoencoder"):

    # shape = (width, height, depth, 1)
    print("\n"+f"{modality.model_name} - compile unet model".center(50, '.'))
    #keras.backend.clear_session()

    #TODO: sbatch variable for unet backbone
    #autoencoder = sm.Unet("vgg16", input_shape=(dim[0], dim[1], dim[2],3), encoder_weights="imagenet")

    modality.modeltype = "autoencoder"

    if modality.merged:
        autoencoder = get_merged_model()
    else:
        dims = modality.image_shape if isinstance(modality.image_shape, tuple) else modality.reshape_dim
        autoencoder = sm.Unet(
            backbone_name = modality.backbone_name, #"vgg16",
            input_shape = (dims[0],dims[1],dims[2],3),#(None,None,None,3),#modality.input_shape,
            encoder_weights = modality.encoder_weights,# "imagenet",
            classes = modality.classes,# 1,
            encoder_freeze = modality.encoder_freeze,# False,
            decoder_block_type = modality.decoder_block_type,# "transpose",
            activation = modality.activation)# "sigmoid")

        if not os.path.exists(modality.model_path):
            os.makedirs(modality.model_path)


    tf.keras.utils.plot_model(
        autoencoder,
        show_shapes=True,
        show_layer_activations = True,
        expand_nested=True,
        to_file=os.path.abspath(modality.model_path+f"{modality.modeltype}.png")
        )
    #https://github.com/ZFTurbo/segmentation_models_3D/blob/master/segmentation_models_3D/models/unet.py#L166

    #TODO: sbatch variable for learning_rate
    #opt = tf.keras.optimizers.Adam()
    #learning_rate = opt.lr.numpy()*len(tf.config.list_physical_devices('GPU'))
    #opt.lr.assign(learning_rate)

    #autoencoder._name = f"{modality_name}_{os.environ['SLURM_JOB_NAME']}"
    autoencoder._name = f"{modality.modeltype}_{modality.modality_name}"#f"{modality_name}_{modality.job_name}"

    #TODO: check loss function?
    #TODO: add MSE and MAE for each epoch and save log
    #TODO: add loss and val loss function for each epoch

    #reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE #SUM #NONE 
    #https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanAbsoluteError
    #mse = tf.keras.losses.MeanSquaredError()#reduction)
    #mae = tf.keras.losses.MeanAbsoluteError()#reduction)
    


    opt = tf.keras.optimizers.Adam(learning_rate=modality.autoencoder_learning_rate)
    loss = tf.keras.losses.MeanSquaredError(name="loss")
    metrics = [
        tf.keras.metrics.MeanSquaredError(name='MSE'),
        tf.keras.metrics.MeanAbsoluteError(name='MAE')
    ]

    autoencoder.compile(opt, loss=loss, metrics=metrics)


    print(autoencoder.summary())

    

    return autoencoder



def model_building(trainDS, valDS):
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
    
    model = get_unet_model(modality.modality_name)



    #TODO: load selected weights from sbatch variable if exists
    
    #model = train_model(model,x_data[idx], y_data[idx])
    #savepath = f"../models/{os.environ['SLURM_JOB_NAME']}/{modality}"
    #savepath = f"../models/{os.environ['SLURM_JOB_NAME']}/{modality}/{os.environ['SLURM_JOB_ID']}/"
    #save_path = os.path.join(modality.model_path,modality.model_name)
    #with modality.strategy.scope():
    # if os.path.isdir(modality.model_path): 
    #     #if len(gpus)>1:
    #     #    with strategy.scope():
    #             # model.load_weights(os.path.join(savepath,"weights"))    
    #             #model.load_weights(os.path.join(savepath,modality)) #savepath)
    #     #        model.load_weights(save_path)
    #     #        print("Loaded Weights")
    #     #else: #model.load_weights(os.path.join(savepath,modality)) 
    #     #    model.load_weights(save_path)       
    #     #with modality.strategy.scope:
    #     model.load_weights(save_path)
    #     print("Loaded Weights")
    # else: 
    #     model = train_model(model, trainDS, valDS)
        #model.save(save_path)   
        #model.save_weights(os.path.join(savepath,modality))

    model = train_model(model, trainDS, valDS)

    #TODO: save images from recreation, with noisy and clean included (use more than 5 images in the plot)
        
    # model.save(f"../models/{model.name}/{os.environ['SLURM_JOB_NAME']}/{os.environ['SLURM_JOB_ID']}-{os.environ['SLURM_JOB_NAME']}")

    print("\n"+f" Model building finished {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

    return model


def build_train_classifier(encoder, y_train, y_val, labels):
    
    modality.modeltype = "classifier"

    
    if modality.classifier_freeze_encoder:
        for layer in encoder.layers:
            if not layer.name.startswith("center"):
                layer.trainable = False

    x = encoder.output
    x = layers.Flatten(name='flatten')(x)
    
    if modality.classifier_multi_dense:
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
    x = layers.Dense(1, activation='sigmoid', name='predictions')(x)

    #N = x.shape[-1]
    
    classifier = Model(encoder.input, x,name=f'{modality.modeltype}_{modality.modality_name}')
    
    if encoder.input.shape[-1] != 1:
        inp = layers.Input(shape=(*encoder.input.shape[1:-1],1))
        l1 = layers.Conv3D(3, (1, 1, 1))(inp) # map N channels data to 3 channels
        out = classifier(l1)
        classifier = Model(inp, out, name=f'{modality.modeltype}_{modality.modality_name}')

    opt = tf.keras.optimizers.Adam(learning_rate=modality.classifier_train_learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False,name="BinaryCrossentropy_loss")
    metrics = tf.keras.metrics.AUC(num_thresholds=200, name = "ROC_AUC", curve="ROC")
    classifier.compile(opt, loss=loss, metrics=metrics)

    
    tf.keras.utils.plot_model(
        classifier,
        show_shapes=True,
        show_layer_activations = True,
        expand_nested=True,
        to_file=os.path.abspath(modality.model_path+f"{modality.modeltype}.png")
        )


    if isinstance(y_train, np.ndarray):
        train_loader = tf.data.Dataset.from_tensor_slices(({f"input_{i+1}_{modality.merged_modalities[i]}":y for i,y in enumerate(y_train)},labels["y_train"]))
        val_loader = tf.data.Dataset.from_tensor_slices(({f"input_{i+1}_{modality.merged_modalities[i]}":y for i,y in enumerate(y_val)},labels["y_val"]))

    else:
        train_loader = tf.data.Dataset.from_tensor_slices((y_train, labels["y_train"]))
        val_loader = tf.data.Dataset.from_tensor_slices((y_val, labels["y_val"]))
    
    trainDS = (
        train_loader
            .batch(
                batch_size = modality.classifier_train_batchsize
                ,num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(
                buffer_size = tf.data.AUTOTUNE)
                )
    valDS = (
        val_loader
            .batch(
                batch_size = modality.classifier_train_batchsize
                ,num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(
                buffer_size = tf.data.AUTOTUNE)
                )

    #classifier = train_classifier(classifier, trainDS, valDS)

    print("\n"+f"{modality.modeltype}-{modality.model_name} - training started".center(50, '.'))      
                    
    pltcallback = PlotCallback()
    pltcallback.x_val = None

    classifier.fit(
        trainDS,
        validation_data = valDS,
        epochs = modality.classifier_train_epochs,
        verbose=2,
        callbacks = [pltcallback],
    )
    
    classifier.save(modality.model_path+"/classifier/")
    
    return classifier


def stats(r):
        alpha = 0.95
        p = [((1.0-alpha)/2.0) *100,(alpha+((1.0-alpha)/2.0))* 100]
        average = sum(r) / len(r)
        std = statistics.stdev(r)
        # confidence intervals
        lower_boot = max(0.0,np.percentile(r,p[0]))
        upper_boot = min(1.0,np.percentile(r,p[1]))
        return average,std,lower_boot,upper_boot


def evaluate_classifier(classifier, y_test, labels):
    
    modality.modeltype = "evaluate_classifier"


    opt = tf.keras.optimizers.Adam(learning_rate=modality.classifier_test_learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False,name="BinaryCrossentropy_loss")
    metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            #tfa.metrics.F1Score(num_classes=1, average=None),
            tf.keras.metrics.AUC(num_thresholds=200, name = "ROC_AUC", curve="ROC"),
            tf.keras.metrics.AUC(num_thresholds=200, name = "PR_AUC", curve="PR"),]

    classifier.compile(opt, loss=loss, metrics=metrics)




    # print(cfr_dataframe)

    # print(cfr)

    # modality.cfr = cfr

    # print(modality.mrkdown())


    #auc_l = list()
    #if isinstance(y_test, np.ndarray):
    #    n_size = int(y_test[0].shape[0])
    #else:
    #    n_size = int(y_test.shape[0])
    n_size = len(labels)
    n_iterations = modality.classifier_test_Nbootstrap

    # print(y_test.shape)
    # print(len(labels))
    # print(labels.shape)
    names = [loss.name,*[metrics[i].name for i in range(len(metrics))],"f1_score"]
    results = {n:list() for n in names}
    results_stats = {n:{f"{s}_{n}":list() for s in ["avg","std","lower","upper"]} for n in names}
    

    test_writer = tf.summary.create_file_writer(logdir=os.path.join(modality.tensorboard_path,f"{modality.modeltype}_test"))

    for i in range(1,n_iterations+1):
        #a = np.random.randint(1,20000)
        if isinstance(y_test, np.ndarray):
            test = [resample(y.numpy(),n_samples= n_size, replace = True,stratify = labels, random_state = i) for y in y_test]
        else:
            test = resample(y_test.numpy(),n_samples= n_size, replace = True,stratify = labels, random_state = i)
            
        
        labels_test = resample(labels.numpy(),n_samples = n_size, replace = True,stratify = labels, random_state = i)
        
        res = classifier.evaluate(test,labels_test,batch_size = modality.classifier_test_batchsize, verbose = 0)

        #auc_l.append(res[-1])
        #print(f"Run: {i}/{n_iterations} - AUC: {res[-1]}")
        
        results = {}
        for n,r in enumerate(res):
            results[names[n]].append(r)

        results["f1_score"].append((2*results["precision"]*results["recall"])/(results["precision"]+results["recall"]))
            
        if i >1:
            
            # stat = {}
            # for k,v in results.items():
            #     s = stats(v)
            #     for i,rsk in enumerate(results_stats[k].keys()):
            #         results_stats[k][rsk].append(s[i])
            #     #stat[f"avg_{k}"], stat[f"std_{k}"], stat[f"lower_{k}"], stat[f"upper_{k}"] = stats(results[k])
                
            
            with test_writer.as_default():
                for key, value in results.items():
                    #tf.summary.scalar(f"{modality.modality_name}/{modality.modeltype}_{key}",value[-1],i)
                    tf.summary.scalar(f"{modality.modeltype}/{key}",value[-1],i)
                    s = stats(value)
                    for i,k in enumerate(results_stats[key].keys()):
                        results_stats[key][k].append(s[i])
                    #for k, v in results_stats[key].items():
                        #tf.summary.scalar(f"{modality.modality_name}/{modality.modeltype}_{k}",s[i],i)
                        tf.summary.scalar(f"{modality.modeltype}/{k}",s[i],i)
                test_writer.flush()

            print(f"\n{i}/{n_iterations}")
            [print([f'{key} {value[-1]:.1f} - {" - ".join([f"{k[:-len(key)-1]} {v[-1]:.1f}"for k, v in results_stats[key].items()])}'][0])for key, value in results.items()][0]

        del results, test, labels_test



    prediction = classifier.predict([y_test],batch_size = modality.classifier_test_batchsize)
    
    cfr = classification_report(labels.numpy(),np.argmax(prediction,axis=1),target_names=["non-significant","significant"], output_dict=True)   

    figure = plt.figure()
    sns.set(font_scale=1.2)
    sns.heatmap(pd.DataFrame(cfr).T, annot=True, annot_kws={"size": 16})
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    

    modality.results = results
    modality.results_stats = results_stats    
    modality.cfr = cfr
    description = modality.mrkdown()

    with test_writer.as_default():
            tf.summary.image(modality.job_name+f"_img/conf_{modality.modality_name}_{modality.modeltype}", image, step=i, description=description)
            test_writer.flush()

    with test_writer.as_default():
        tf.summary.text(modality.job_name+f"_txt/{modality.modality_name}_{modality.modeltype}", description, step=n_iterations+1, description=modality.model_name)
        test_writer.flush()


    with test_writer.as_default():
        tf.summary.experimental.write_raw_pb(
            create_layout_summary(results,results_stats,modality.modeltype).SerializeToString(), step=0
        )



def create_layout_summary(results,results_stats,model_type):
    return cs_summary.pb(
        layout_pb2.Layout(
            category=[[
                layout_pb2.Category(
                    title=key,
                    chart=[
                            layout_pb2.Chart(
                                title=key,
                                margin=layout_pb2.MarginChartContent(
                                    series=[
                                        layout_pb2.MarginChartContent.Series(
                                            value=f"{model_type}/{key}",
                                            lower=f"{model_type}/{[k_ for k_ in results_stats[key].keys() if k_.startswith('lower')][0]}",
                                            upper=f"{model_type}/{[k_ for k_ in results_stats[key].keys() if k_.startswith('upper')][0]}",
                                        ),
                                    ],
                                ),
                            ),
                        [
                        layout_pb2.Chart(
                            title=k,
                            margin=layout_pb2.MarginChartContent(
                                series=[
                                    layout_pb2.MarginChartContent.Series(
                                        value=f"{model_type}/{k}",
                                        lower=f"{model_type}/{[k_ for k_ in results_stats[key].keys() if k_.startswith('lower')][0]}",
                                        upper=f"{model_type}/{[k_ for k_ in results_stats[key].keys() if k_.startswith('upper')][0]}",
                                    ),
                                ],
                            ),
                        ),
                    ]for k in results_stats[key].keys() if not k.startswith("upper") or not k.startswith("lower")
                    ],
                )
            ]for key in results.keys()],
        ),
    )

