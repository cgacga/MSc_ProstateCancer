
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
        
        self.epoch = 0
        
        self.savename = f"{modality.modality_name}_{self.modeltype}"
        

        self.epoch_modulo = modality.tensorboard_img_epoch
        self.nimages = modality.tensorboard_num_predictimages
        self.job_name = modality.job_name
        self.description = modality.mrkdown()
        self.model_name = modality.model_name
        self.modality_name = modality.modality_name

        self.train_writer = tf.summary.create_file_writer(logdir=os.path.join(modality.tensorboard_path,f"{self.modeltype}_train"))
        self.val_writer = tf.summary.create_file_writer(logdir=os.path.join(modality.tensorboard_path,f"{self.modeltype}_val"))
        #self.file_writer = self.val_writer

        
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
            #img = self.model.predict([tf.expand_dims(xval[i],0) for xval in self.__class__.x_val])
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


        if train_logs:
            with self.train_writer.as_default():
                for key, value in train_logs.items():
                    for name in modality.merged_modalities:
                        if key.startswith(name):
                            tf.summary.scalar(f"{name}/{self.modality_name}_{self.modeltype}_{key[len(name)+1:]}",value,self.epoch)
                    if key.startswith("loss"):
                        tf.summary.scalar(f"{self.modality_name}/{self.modeltype}_global_{key}",value,self.epoch)
                    else:
                        tf.summary.scalar(f"{self.modality_name}/{self.modeltype}_{key}",value,self.epoch)


                self.train_writer.flush()
        if val_logs:
            with self.val_writer.as_default():
                for key, value in val_logs.items():
                    key = key[4:]
                    for name in modality.merged_modalities:
                        if key.startswith(name):
                            tf.summary.scalar(f"{name}/{self.modality_name}_{self.modeltype}_{key[len(name)+1:]}",value,self.epoch)
                    if key.startswith("loss"):
                        tf.summary.scalar(f"{self.modality_name}/{self.modeltype}_global_{key}",value,self.epoch)#
                    else:
                        tf.summary.scalar(f"{self.modality_name}/{self.modeltype}_{key}",value,self.epoch)#


                self.val_writer.flush()

        
        
    def on_epoch_end(self, epoch, logs={}):           
        self.epoch = epoch+1     


        
        
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
    
    print("\n"+f"{modality.model_name} - training started".center(50, '.'))

    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)


    #x_val = list(valDS.map(lambda x,y: (x[0:modality.tensorboard_num_predictimages])))[0]
    #x_val = list(valDS.as_numpy_iterator())[0][0][0:modality.tensorboard_num_predictimages]
    pltcallback = PlotCallback()

    model.fit(
        # x=train_data,
        trainDS,
        validation_data = valDS,
        epochs = modality.autoencoder_epocs,
        #batch_size=batch_size,
        verbose=2,
        
        # callbacks=[checkpoint_cb, early_stopping_cb],
        #callbacks=early_stopping_cb,
        #steps_per_epoch =  modality.steps_pr_epoch,
        #validation_steps =  modality.validation_steps,
        callbacks = [pltcallback],
        #callbacks = [tensorboard_callback,pltcallback],
    )


    return model

def get_unet_model(modality_name="autoencoder"):

    # shape = (width, height, depth, 1)
    print("\n"+f"{modality.model_name} - compile unet model".center(50, '.'))
    #keras.backend.clear_session()

    #TODO: sbatch variable for unet backbone
    #autoencoder = sm.Unet("vgg16", input_shape=(dim[0], dim[1], dim[2],3), encoder_weights="imagenet")

    modality.modeltype = "autoencoder"
    modality.results = modality.results_stats = modality.cfr = None

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
            activation = modality.activation,
            decoder_use_batchnorm = modality.batchnorm,
            dropout = modality.dropout)# "sigmoid")

    if modality.self_superviced:
        if not os.path.exists(modality.AE_path):
            os.makedirs(modality.AE_path)

        tf.keras.utils.plot_model(
            autoencoder,
            show_shapes=True,
            show_layer_activations = True,
            expand_nested=True,
            to_file=os.path.abspath(modality.AE_path+f"{modality.modeltype}.png")
            )
    #https://github.com/ZFTurbo/segmentation_models_3D/blob/master/segmentation_models_3D/models/unet.py#L166

    autoencoder._name = f"{modality.modeltype}_{modality.modality_name}"

    opt = tf.keras.optimizers.Adam(learning_rate=modality.autoencoder_learning_rate)
    loss = tf.keras.losses.MeanSquaredError(name="loss_mse")
    metrics = [
        tf.keras.metrics.MeanSquaredError(name='mse'),
        tf.keras.metrics.MeanAbsoluteError(name='mae')
    ]

    autoencoder.compile(opt, loss=loss, metrics=metrics)


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


    model = get_unet_model(modality.modality_name)

    if modality.self_superviced:
        model = train_model(model, trainDS, valDS)


    print("\n"+f" Model building finished {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '_')+"\n")

    return model


def build_train_classifier(encoder, y_train, y_val, labels):
    
    modality.modeltype = "classifier"
    modality.results = modality.results_stats = modality.cfr = None

    
    if modality.classifier_freeze_encoder:
        for layer in encoder.layers:
            if not layer.name.startswith("center"):
                layer.trainable = False

    x = encoder.output
    x = layers.Flatten(name='flatten')(x)
    
    if modality.classifier_multi_dense:
        #x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.Dense(1, activation='sigmoid', name='predictions')(x)

    #N = x.shape[-1]
    
    classifier = Model(encoder.input, x,name=f'{modality.modeltype}_{modality.modality_name}')
    
    if isinstance(encoder.input,list):
        input_layers = []
        conv_layers = []
        if any([inp.shape[-1] >1 for inp in encoder.input]):
            for in_ in encoder.input:
                input_ = layers.Input(shape=(*in_.shape[1:-1],1), name =in_.name)
                input_layers.append(input_)
                conv_layers.append(layers.Conv3D(3, (1, 1, 1))(input_))
            
            classifier = Model(input_layers,classifier(conv_layers))
    else:
        if encoder.input.shape[-1] != 1:
            inp = layers.Input(shape=(*encoder.input.shape[1:-1],1))
            l1 = layers.Conv3D(3, (1, 1, 1))(inp) # map N channels data to 3 channels
            out = classifier(l1)
            classifier = Model(inp, out, name=f'{modality.modeltype}_{modality.modality_name}')

    opt = tf.keras.optimizers.Adam(learning_rate=modality.classifier_train_learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False,name="BinaryCrossentropy_loss")
    metrics = tf.keras.metrics.AUC(num_thresholds=200, name = "ROC_AUC", curve="ROC")
    classifier.compile(opt, loss=loss, metrics=metrics)




    if isinstance(y_train, np.ndarray):
        train_loader = tf.data.Dataset.from_tensor_slices(({f"input_{i+1}":y for i,y in enumerate(y_train)},labels["y_train"]))
        val_loader = tf.data.Dataset.from_tensor_slices(({f"input_{i+1}":y for i,y in enumerate(y_val)},labels["y_val"]))

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
    
    classifier.save(modality.C_path)


    tf.keras.utils.plot_model(
        classifier,
        show_shapes=True,
        show_layer_activations = True,
        expand_nested=True,
        to_file=os.path.abspath(modality.C_path+f"{modality.modeltype}.png")
        )
    
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
    modality.results = modality.results_stats = modality.cfr = None


    opt = tf.keras.optimizers.Adam(learning_rate=modality.classifier_test_learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False,name="BinaryCrossentropy_loss")
    metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            #tfa.metrics.F1Score(num_classes=1, average=None),
            tf.keras.metrics.AUC(name = "ROC_AUC", curve="ROC"),
            tf.keras.metrics.AUC(name = "PR_AUC", curve="PR")]

    classifier.compile(opt, loss=loss, metrics=metrics)



    n_size = len(labels)
    n_iterations = modality.Nbootstrap_steps

    names = [loss.name,*[metrics[i].name for i in range(len(metrics))],"f1_score"]
    results = {n:list() for n in names}
    results_stats = {n:{f"{s}_{n}":list() for s in ["avg","std","lower","upper"]} for n in names}
    

    evaluate_writer = tf.summary.create_file_writer(logdir=os.path.join(modality.tensorboard_path,f"{modality.modeltype}_eval"))

    for i in range(1,n_iterations+1):
        #a = np.random.randint(1,20000)
        if isinstance(y_test, np.ndarray):
            test = [resample(y.numpy(),n_samples= n_size, replace = True,stratify = labels, random_state = i) for y in y_test]
        else:
            test = resample(y_test.numpy(),n_samples= n_size, replace = True,stratify = labels, random_state = i)
            
        
        labels_test = resample(labels.numpy(),n_samples = n_size, replace = True,stratify = labels, random_state = i)
        
        res = classifier.evaluate(test,labels_test,batch_size = modality.classifier_test_batchsize, verbose = 0)

        for n,r in enumerate(res):
            results[names[n]].append(r)
        try:
            results["f1_score"].append((2*results["precision"][-1]*results["recall"][-1])/(results["precision"][-1]+results["recall"][-1]))
        except:
            pass
            
        if i >1:
            
            
            with evaluate_writer.as_default():
                for key, value in results.items():                  

                    if key == "f1_score":
                      if len(results["f1_score"]) <2:
                          continue
                    tf.summary.scalar(f"{modality.modeltype}/{key}",value[-1],i)
                    s = stats(value)
                    for index,k in enumerate(results_stats[key].keys()):
                        results_stats[key][k].append(s[index])
                    #for k, v in results_stats[key].items():
                        #tf.summary.scalar(f"{modality.modality_name}/{modality.modeltype}_{k}",s[i],i)
                        tf.summary.scalar(f"{modality.modeltype}/{k}",s[index],i)
                evaluate_writer.flush()

            print(f"\n{i}/{n_iterations}")
            try:
                [print([f'{key} {value[-1]:.2f} - {" - ".join([f"{k[:-len(key)-1]} {v[-1]:.2f}"for k, v in results_stats[key].items()])}'][0])for key, value in results.items()][0]
            except:
                pass



            
        if isinstance(y_test, np.ndarray):
            prediction = classifier.predict([y for y in test],batch_size = modality.classifier_test_batchsize)
        else:
            prediction = classifier.predict([test],batch_size = modality.classifier_test_batchsize)
        
        #cfr = classification_report(labels.numpy(),np.argmax(prediction,axis=1),target_names=["non-significant","significant"], output_dict=True)   


        cfr = classification_report(labels_test,np.round(prediction),zero_division=0, output_dict=True)   

        figure = plt.figure()
        sns.set(font_scale=0.8)
        sns.heatmap(pd.DataFrame(cfr).T, annot=True, annot_kws={"size": 14}, vmin=0.0, vmax=1.0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)


        with evaluate_writer.as_default():
                tf.summary.image(modality.job_name+f"_img/conf_{modality.modality_name}_{modality.modeltype}", image, step=i)
                evaluate_writer.flush()

        print(classification_report(labels_test,np.round(prediction),target_names=["non-significant","significant"]))


        del test, labels_test, res
    

    modality.results = results
    modality.results_stats = results_stats    
    modality.cfr = cfr
    description = modality.mrkdown()


    with evaluate_writer.as_default():
        tf.summary.text(modality.job_name+f"_txt/{modality.modality_name}_{modality.modeltype}", description, step=n_iterations+1, description=modality.model_name)
        evaluate_writer.flush()


    with evaluate_writer.as_default():
        tf.summary.experimental.write_raw_pb(
            create_layout_summary(results,results_stats,modality.modeltype).SerializeToString(), step=0
        )


def create_layout_summary(results,results_stats,model_type):
    charts = [[
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
                        ),*[[
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
                    ][0] for k in results_stats[key].keys() if not k.startswith("upper") and not k.startswith("lower")]],
                )
            ][0] for key in results.keys()]
    a,b,c,d,e,f,g = charts
    return cs_summary.pb(layout_pb2.Layout(
            category=[a,b,c,d,e,f,g]
        ))

