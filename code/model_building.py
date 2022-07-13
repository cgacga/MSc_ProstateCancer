
### Model building ###
import os, time, json, statistics, gc
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
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from tensorboard.plugins.custom_scalar import summary as cs_summary
from tensorboard.plugins.custom_scalar import layout_pb2
import seaborn as sns

class PlotCallback(tf.keras.callbacks.Callback):
    x_val = None
    x_val_temp = None

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

        
    def plot_predictions(self):
        
        
        pred_sample = np.zeros_like(self.__class__.x_val)
        for i,xval in enumerate(self.__class__.x_val):
            pred_sample[i] = self.model.predict(tf.repeat(tf.expand_dims(xval,0),3,-1))
            
        
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
            img_stack = np.stack([self.__class__.x_val_temp[i],self.__class__.x_val[i],pred_sample[i]],0+1).reshape(*(-(0==j) or s for j,s in enumerate(pred_sample[i].shape)))
            
            for i,all in enumerate([True,False]):
                image = img_pltsave([img for img in img_stack],None,True,all)
                with self.val_writer.as_default():
                    tf.summary.image(self.job_name+f"_img{i+1}/{name}_{self.savename}", image, step=self.epoch, description=self.description)
                    self.val_writer.flush()


    def json_summary(self,logs):
        modality.training_logs = logs
        modality.epoch_stop = self.epoch
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
    
    print("\n"+f"{modality.model_name} - training started".center(50, '.'))

    pltcallback = PlotCallback()

    model.fit(
        trainDS,
        validation_data = valDS,
        epochs = modality.autoencoder_epocs,
        batch_size=modality.autoencoder_batchsize,
        verbose=2,
        # callbacks=[checkpoint_cb, pltcallback],
        callbacks = [pltcallback]
    )


    return model

def get_unet_model(modality_name="autoencoder"):

    print("\n"+f"{modality.model_name} - compile unet model".center(50, '.'))
    
    modality.modeltype = "autoencoder"
    modality.results = modality.results_stats = modality.cfr = None

    if modality.merged:
        autoencoder = get_merged_model()
    else:
        dims = modality.image_shape if isinstance(modality.image_shape, tuple) else modality.reshape_dim
        autoencoder = sm.Unet(
            backbone_name = modality.backbone_name, 
            input_shape = (dims[0],dims[1],dims[2],3),
            encoder_weights = modality.encoder_weights,
            classes = modality.classes,
            encoder_freeze = modality.encoder_freeze,
            decoder_block_type = modality.decoder_block_type,
            activation = modality.activation,
            decoder_use_batchnorm = modality.batchnorm,
            dropout = modality.dropout)

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

    x = encoder.output
    x = layers.Flatten(name='cls_flatten')(x)
    
    if modality.classifier_multi_dense:
        x = layers.Dense(4096, activation='relu', name='cls_fc1')(x)
        x = layers.Dense(4096, activation='relu', name='cls_fc2')(x)
    x = layers.Dense(2, activation='softmax', name='cls_predictions')(x)

    classifier = Model(encoder.input, x,name=f'{modality.modeltype}_{modality.modality_name}')
    
    if modality.classifier_freeze_encoder:
        for layer in classifier.layers:
            #if not layer.name.startswith(("center","cls_")):
            if not layer.name.startswith(("cls_")):
                layer.trainable = False
    
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
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False,name="clstrain_BinaryCrossentropy_loss")
    metrics = tf.keras.metrics.AUC(multi_label=False, name = "clstrain_ROC_AUC", curve="ROC")
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

    

    print("\n"+f"{modality.modeltype}-{modality.model_name} - training started".center(50, '.'))      
                    
    pltcallback = PlotCallback()
    pltcallback.x_val = None

    #early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

    classifier.fit(
        trainDS,
        validation_data = valDS,
        epochs = modality.classifier_train_epochs,
        verbose=2,
        callbacks = [pltcallback]
        #callbacks = [pltcallback,early_stopping_cb]
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
        # CI
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
            tf.keras.metrics.AUC(name = "ROC_AUC", curve="ROC")
            ]

    classifier.compile(opt, loss=loss, metrics=metrics)



    n_size = len(labels)
    n_iterations = modality.Nbootstrap_steps

    #names = [loss.name,*[metrics[i].name for i in range(len(metrics))],"f1_score"]
    names = [loss.name,*[metrics[i].name for i in range(len(metrics))],"tn", "fp", "fn", "tp",
    "accuracy", "precision_0", "precision_1", "precision_macro", "precision_w_macro", "recall_0", "recall_1", "recall_macro", "recall_w_macro", "f1_0", "f1_1", "f1_macro", "f1_w_macro", "specificity_0", "specificity_1", "specificity_macro", "specificity_w_macro"]
    results = {n:list() for n in names}
    results_stats = {n:{f"{s}_{n}":list() for s in ["avg","std","lower","upper"]} for n in names}
    

    evaluate_writer = tf.summary.create_file_writer(logdir=os.path.join(modality.tensorboard_path,f"{modality.modeltype}_eval"))
    support_0,support_1,support = 0,0,0
    
    for i in range(1,n_iterations+1):
        #a = np.random.randint(1,20000)
        if isinstance(y_test, np.ndarray):
            test = [resample(y.numpy(),n_samples= n_size, replace = True,stratify = labels, random_state = i) for y in y_test]
        else:
            test = resample(y_test.numpy(),n_samples= n_size, replace = True,stratify = labels, random_state = i)
            
        
        labels_test = resample(labels.numpy(),n_samples = n_size, replace = True,stratify = labels, random_state = i)
        
        res = classifier.evaluate(test,labels_test,batch_size = modality.classifier_test_batchsize, verbose = 0)

        for n,r in enumerate(res):
            results[names[n]].append(np.float32(r))
                    
        if isinstance(y_test, np.ndarray):
            prediction = classifier.predict([y for y in test],batch_size = modality.classifier_test_batchsize)
        else:
            prediction = classifier.predict([test],batch_size = modality.classifier_test_batchsize)

        y_pred_bool = np.argmax(prediction, axis=1)
        labels_bool = np.argmax(labels_test, axis=1)

        tn, fp, fn, tp = confusion_matrix(labels_bool, y_pred_bool).ravel()

        cfr = classification_report(labels_bool,y_pred_bool,zero_division=0, output_dict=True) 


        results["tn"].append(tn)
        results["tp"].append(tp)
        results["fn"].append(fn)
        results["fp"].append(fp)

        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # f1 = 2 * precision * recall / (precision + recall)


        if i <=1:
            support_0 = cfr["0"]["support"]
            support_1 = cfr["1"]["support"]
            support = cfr["macro avg"]["support"]
        
        results["accuracy"].append(np.float32(cfr["accuracy"]))

        results["precision_0"].append(np.float32(cfr["0"]["precision"]))
        results["precision_1"].append(np.float32(cfr["1"]["precision"]))
        results["precision_macro"].append(np.float32(cfr["macro avg"]["precision"]))
        results["precision_w_macro"].append(np.float32(cfr["weighted avg"]["precision"]))

        results["recall_0"].append(np.float32(cfr["0"]["recall"]))
        results["recall_1"].append(np.float32(cfr["1"]["recall"]))
        results["recall_macro"].append(np.float32(cfr["macro avg"]["recall"]))
        results["recall_w_macro"].append(np.float32(cfr["weighted avg"]["recall"]))

        results["f1_0"].append(np.float32(cfr["0"]["f1-score"]))
        results["f1_1"].append(np.float32(cfr["1"]["f1-score"]))
        results["f1_macro"].append(np.float32(cfr["macro avg"]["f1-score"]))
        results["f1_w_macro"].append(np.float32(cfr["weighted avg"]["f1-score"]))

        specificity_0 = tp/(tp+fn)
        specificity_1 = tn/(tn+fp)

        results["specificity_0"].append(np.float32(specificity_0))
        results["specificity_1"].append(np.float32(specificity_1))
        results["specificity_macro"].append(np.float32((specificity_0+specificity_1)/2))
        results["specificity_w_macro"].append(np.float32(((specificity_0*support_0)+(specificity_1*support_1))/support))


            
        if i >1:
            
            
            with evaluate_writer.as_default():
                for key, value in results.items():                  
                    

                    if key.startswith(("BinaryCrossentropy_loss","ROC_AUC","accuracy","f1_macro")):
                        tf.summary.scalar(f"{modality.modeltype}/{key}",value[-1],i)

                    s = stats(value)
                    for index,k in enumerate(results_stats[key].keys()):
                        if key.startswith(("BinaryCrossentropy_loss","ROC_AUC","accuracy","f1_macro")):
                            results_stats[key][k].append(np.float32(s[index]))
                            tf.summary.scalar(f"{modality.modeltype}/{k}",s[index],i)
                        else:
                            results_stats[key][k] = np.float32(s[index])
                evaluate_writer.flush()

            # Print
            print(f"\n{i}/{n_iterations}")
            try:
                [print([f'{key} {value[-1]:.2f} - {" - ".join([f"{k[:-len(key)-1]} {v[-1]:.2f}"  if isinstance(v, list) else f"{k[:-len(key)-1]} {v:.2f}" for k, v in results_stats[key].items()])}'][0])for key, value in results.items()][0]
            except:
                pass
            cfm = confusion_matrix(labels_bool, y_pred_bool)
            print("Confusion Matrix\n", cfm)


        del test, labels_test, res, specificity_0, specificity_1, y_pred_bool, labels_bool, cfr, tn, fp, fn, tp, prediction
        #tf.keras.backend.clear_session()
        gc.collect()

            
    report = {
        
        "0":{
            "precision":results_stats["precision_0"]["avg_precision_0"],
            "recall":results_stats["recall_0"]["avg_recall_0"],
            "f1":results_stats["f1_0"]["avg_f1_0"],
            "specificity":results_stats["specificity_0"]["avg_specificity_0"],
            "support":support_0},

        "1":{
            "precision":results_stats["precision_1"]["avg_precision_1"],
            "recall":results_stats["recall_1"]["avg_recall_1"],
            "f1":results_stats["f1_1"]["avg_f1_1"],
            "specificity":results_stats["specificity_1"]["avg_specificity_1"],
            "support":support_1},

        "macro avg":{
            "precision":results_stats["precision_macro"]["avg_precision_macro"],
            "recall":results_stats["recall_macro"]["avg_recall_macro"],
            "f1":results_stats["f1_macro"]["avg_f1_macro"][-1],
            "specificity":results_stats["specificity_macro"]["avg_specificity_macro"],
            "support":support},

        "weighted avg":{
            "precision":results_stats["precision_w_macro"]["avg_precision_w_macro"],
            "recall":results_stats["recall_w_macro"]["avg_recall_w_macro"],
            "f1":results_stats["f1_w_macro"]["avg_f1_w_macro"],
            "specificity":results_stats["specificity_w_macro"]["avg_specificity_w_macro"],
            "support":support},
    }

    labels_annot = {

        "0":{
            "precision":f'{results_stats["precision_0"]["avg_precision_0"]:.2f}\n[{results_stats["precision_0"]["lower_precision_0"]:.2f} - {results_stats["precision_0"]["upper_precision_0"]:.2f}]',

            "recall":f'{results_stats["recall_0"]["avg_recall_0"]:.2f}\n[{results_stats["recall_0"]["lower_recall_0"]:.2f} - {results_stats["recall_0"]["upper_recall_0"]:.2f}]',

            "f1":f'{results_stats["f1_0"]["avg_f1_0"]:.2f}\n[{results_stats["f1_0"]["lower_f1_0"]:.2f} - {results_stats["f1_0"]["upper_f1_0"]:.2f}]',

            "specificity":f'{results_stats["specificity_0"]["avg_specificity_0"]:.2f}\n[{results_stats["specificity_0"]["lower_specificity_0"]:.2f} - {results_stats["specificity_0"]["upper_specificity_0"]:.2f}]',

            "support":f'{support_0}'
        },

        "1":{
            "precision":f'{results_stats["precision_1"]["avg_precision_1"]:.2f}\n[{results_stats["precision_1"]["lower_precision_1"]:.2f} - {results_stats["precision_1"]["upper_precision_1"]:.2f}]',

            "recall":f'{results_stats["recall_1"]["avg_recall_1"]:.2f}\n[{results_stats["recall_1"]["lower_recall_1"]:.2f} - {results_stats["recall_1"]["upper_recall_1"]:.2f}]',

            "f1":f'{results_stats["f1_1"]["avg_f1_1"]:.2f}\n[{results_stats["f1_1"]["lower_f1_1"]:.2f} - {results_stats["f1_1"]["upper_f1_1"]:.2f}]',

            "specificity":f'{results_stats["specificity_1"]["avg_specificity_1"]:.2f}\n[{results_stats["specificity_1"]["lower_specificity_1"]:.2f} - {results_stats["specificity_1"]["upper_specificity_1"]:.2f}]',

            "support":f'{support_1}'
        },

        "macro avg":{
            "precision":f'{results_stats["precision_macro"]["avg_precision_macro"]:.2f}\n[{results_stats["precision_macro"]["lower_precision_macro"]:.2f} - {results_stats["precision_macro"]["upper_precision_macro"]:.2f}]',

            "recall":f'{results_stats["recall_macro"]["avg_recall_macro"]:.2f}\n[{results_stats["recall_macro"]["lower_recall_macro"]:.2f} - {results_stats["recall_macro"]["upper_recall_macro"]:.2f}]',

            "f1":f'{results_stats["f1_macro"]["avg_f1_macro"][-1]:.2f}\n[{results_stats["f1_macro"]["lower_f1_macro"][-1]:.2f} - {results_stats["f1_macro"]["upper_f1_macro"][-1]:.2f}]',

            "specificity":f'{results_stats["specificity_macro"]["avg_specificity_macro"]:.2f}\n[{results_stats["specificity_macro"]["lower_specificity_macro"]:.2f} - {results_stats["specificity_macro"]["upper_specificity_macro"]:.2f}]',

            "support":f'{support}'
        },

        "weighted avg":{
            "precision":f'{results_stats["precision_w_macro"]["avg_precision_w_macro"]:.2f}\n[{results_stats["precision_w_macro"]["lower_precision_w_macro"]:.2f} - {results_stats["precision_w_macro"]["upper_precision_w_macro"]:.2f}]',

            "recall":f'{results_stats["recall_w_macro"]["avg_recall_w_macro"]:.2f}\n[{results_stats["recall_w_macro"]["lower_recall_w_macro"]:.2f} - {results_stats["recall_w_macro"]["upper_recall_w_macro"]:.2f}]',

            "f1":f'{results_stats["f1_w_macro"]["avg_f1_w_macro"]:.2f}\n[{results_stats["f1_w_macro"]["lower_f1_w_macro"]:.2f} - {results_stats["f1_w_macro"]["upper_f1_w_macro"]:.2f}]',

            "specificity":f'{results_stats["specificity_w_macro"]["avg_specificity_w_macro"]:.2f}\n[{results_stats["specificity_w_macro"]["lower_specificity_w_macro"]:.2f} - {results_stats["specificity_w_macro"]["upper_specificity_w_macro"]:.2f}]',

            "support":f'{support}'
            }
    }



    cfm = {
        "Positive (1)":{
    "Positive (1)":results_stats["tp"]["avg_tp"],
    "Negative (0)":results_stats["fn"]["avg_fn"]},
    
        "Negative (0)":{
    "Positive (1)":results_stats["fp"]["avg_fp"],
    "Negative (0)":results_stats["tn"]["avg_tn"]}
    }
    
    cfm_labels = {
        "Positive (1)":{
    "Positive (1)":f'TP\n{results_stats["tp"]["avg_tp"]:.2f}\n\u00B1 {results_stats["tp"]["std_tp"]:.2f}',
    "Negative (0)":f'FN\n{results_stats["fn"]["avg_fn"]:.2f}\n\u00B1 {results_stats["fn"]["std_fn"]:.2f}' 
    },
    
        "Negative (0)":{
    "Positive (1)":f'FP\n{results_stats["fp"]["avg_fp"]:.2f}\n\u00B1 {results_stats["fp"]["std_fp"]:.2f}',
    "Negative (0)":f'TN\n{results_stats["tn"]["avg_tn"]:.2f}\n\u00B1 {results_stats["tn"]["std_tn"]:.2f}' 
    }
    }


    div = {
        "ROC AUC":{
    "avg/\n[CI 95]":results_stats["ROC_AUC"]["avg_ROC_AUC"][-1]},
    
    "Accuracy":{
    "avg/\n[CI 95]":results_stats["accuracy"]["avg_accuracy"][-1]}
    }
    
    div_labels = {
        "ROC AUC":{
    "avg/\n[CI 95]":f'{results_stats["ROC_AUC"]["avg_ROC_AUC"][-1]:.2f}\n[{results_stats["ROC_AUC"]["lower_ROC_AUC"][-1]:.2f} - {results_stats["ROC_AUC"]["upper_ROC_AUC"][-1]:.2f}]'},

    "Accuracy":{
    "avg/\n[CI 95]":f'{results_stats["accuracy"]["avg_accuracy"][-1]:.2f}\n[{results_stats["accuracy"]["lower_accuracy"][-1]:.2f} - {results_stats["accuracy"]["upper_accuracy"][-1]:.2f}]' 
    }
    }

    figure = plt.figure(figsize=(12, 5))

    spec = figure.add_gridspec(ncols=4, nrows=3)
    ax1 = figure.add_subplot(spec[0, 0])
    sns.set(font_scale=0.8)
    sns.heatmap(
            data = pd.DataFrame(div).T,
            annot=pd.DataFrame(div_labels).T,
            annot_kws={"size": 14},
            fmt="s",
            cmap="PuOr",
            linewidths=2, linecolor='white',
            ax=ax1,
            cbar=False)
    
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
    ax1.tick_params(length=0)


    ax2 = figure.add_subplot(spec[1:3, 0])

    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(length=0)
    sns.heatmap(
            data = pd.DataFrame(cfm).T,
            annot=pd.DataFrame(cfm_labels).T,
            annot_kws={"size": 12},
            fmt="s",
            cmap="RdYlBu",
            linewidths=2, linecolor='white',
            ax=ax2,
            cbar=False)


    ax3 = figure.add_subplot(spec[:, 1:4])

    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_position('top')
    ax3.tick_params(length=0)
    sns.heatmap(
            data = pd.DataFrame(report).T,
            annot=pd.DataFrame(labels_annot).T,
            annot_kws={"size": 11},
            vmin=0.0, vmax=1.0,
            fmt="s",
            cmap="viridis",
            linewidths=2, linecolor='white',
            ax=ax3)


    figure.suptitle(f"{(modality.bootpercentage * 100):.0f}% labeled training data")
    
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)


    buf = io.BytesIO()
    #plt.tick_params(labeltop=True)
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)


    with evaluate_writer.as_default():
            tf.summary.image(modality.job_name+f"_img/report_{modality.modality_name}_{modality.modeltype}", image, step=i)
            evaluate_writer.flush()

    if isinstance(y_test, np.ndarray):
        prediction = classifier.predict([y for y in y_test],batch_size = modality.classifier_test_batchsize)
    else:
        prediction = classifier.predict([y_test],batch_size = modality.classifier_test_batchsize)

    y_pred_bool = np.argmax(prediction, axis=1)
    labels_bool = np.argmax(labels, axis=1)
    print("Confusion matrix and classification report for the whole validation set")
    print(confusion_matrix(labels_bool, y_pred_bool))
    print(classification_report(labels_bool,y_pred_bool,zero_division=0) )




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
            ][0] for key in results.keys()  if key.startswith(("BinaryCrossentropy_loss","ROC_AUC","accuracy","f1_macro"))]
    a,b,c,d = charts
    return cs_summary.pb(layout_pb2.Layout(
            #category=[a,b,c,d,e,f,g]
            category=[a,b,c,d]
        ))
    # a,b,c,d,e,f,g = charts   
    # return cs_summary.pb(layout_pb2.Layout(
    #         category=[a,b,c,d,e,f,g]
    #     ))

