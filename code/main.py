#%%
### Importing libraries ###
import os, sys, time, random
import numpy as np
import tensorflow as tf
# from tensorflow.keras import backend as K



from preprocess import *
from data_augmentation import *
from model_building import *
from img_display import *

### GPU Cluster setup ###

# IMPORTANT NOTE FOR USERS RUNNING UiS GPU CLUSTERS:
# This script is made to be used with slurm workload manager. The slurm scheduling system will automatically assign gpus and this script will use all awailable GPU's. This script should not be run outside of slurm wihout changing parameters for building the model and compiling it.



# By setting config.gpu_options.allow_growth to True, Tensorflow will only grab as much GPU-memory as needed. If additional memory is required later in the code, Tensorflow will allocate more memory as needed. This allows the user to run two or three programs on the same GPU.
# tf.keras.backend.clear_session()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# session = tf.compat.v1.Session(config=config)

# tf.compat.v1.keras.backend.set_session(session)

# session = tf.compat.v1.InteractiveSession(config=config)

# K.set_session(session)


# The variable TF_CPP_MIN_LOG_LEVEL sets the debugging level of Tensorflow and controls what messages are displayed onscreen. Defaults value is set to 0, so all logs are shown. Set TF_CPP_MIN_LOG_LEVEL to 1 to filter out INFO logs, 2 to additional filter out WARNING, and 3 to additionally filter out ERROR. Disable debugging information from tensorflow. 
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
# TF_CPP_MIN_VLOG_LEVEL brings in extra debugging information, and in reverse. Its default value is 0 and as it increases, more debugging messages are logged in.
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'


# Deprecation removes deprecated warning messages
# from tensorflow.python.framework import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = True

# Seed for reproducibility
# np.random.seed(42)

# For reproducible results    
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(s) 
seed_all(42)


def main(**kwargs):
    start_time = time.time()
    data_path = "../data/manifest-A3Y4AE4o5818678569166032044/"
    tags = {"ADC": (32,128,96),"t2tsetra": None} 
    #tags = {"t2tsetra": None} 

    y_train, y_val, pat_df = preprocess(data_path,tags)
    # x_train, x_test, x_val, y_train, y_val, y_val = data_augmentation(pat_slices, pat_df)

    # x_train, _, _, x_train_noisy, _, _ = data_augmentation(pat_slices, pat_df)
    # del pat_slices



    


    models = {}
    for modality in tags.keys():
        # modelpath = f"../models/{modality}/{os.environ['SLURM_JOB_NAME']}/{os.environ['SLURM_JOB_ID']}-{os.environ['SLURM_JOB_NAME']}"
        modelpath = f"../models/{modality}/{os.environ['SLURM_JOB_NAME']}"

        shape,idx = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality, case=False)].values[0]
        # train_data = tf.data.Dataset.from_tensor_slices((x_train_noisy[idx], x_train[idx]))
        # val_data = tf.data.Dataset.from_tensor_slices((x_test_noisy[idx], x_test[idx]))
        # models[modality] = model_building(shape, modelpath, train_data, val_data)
        #https://stackoverflow.com/questions/52724022/model-fits-on-a-single-gpu-but-script-crashes-when-trying-to-fit-on-multiple-gpu

        #model = model_building(shape, modelpath, x_train[idx],y_train[idx], x_test[idx], y_val[idx])
        

        trainDS, valDS = augment_build_datasets(y_train[idx], y_val[idx])

        model = model_building(shape, modelpath, trainDS, valDS)
 

        #pred_test = tf.expand_dims(y_train[idx][0],axis=0)
        pred_test = augment_patches(y_val[idx][0])
        
        print("shape staert")
        print(pred_test.shape)
        print("prediction")

        gpus = tf.config.list_physical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy()
        if len(gpus)>1:
            with strategy.scope():
                predictions = model.predict(pred_test)
        else:
            predictions = model.predict(pred_test)
        print("saving img")
        img_pltsave([y_val[idx][0], pred_test, predictions],os.path.join(modelpath,f"y_val_0.png-{modality}"))



        pred_val = augment_patches(y_val[idx][0:5])
        print("shape 0:5")
        print(pred_val.shape)
        print("prediction")
        
        if len(gpus)>1:
            with strategy.scope():
                predictions = model.predict(pred_val)
        else: predictions = model.predict(pred_val)
        print("prediction shape")
        print(predictions.shape)
        print("saving img 05")
        img_pltsave([asd for asd in (np.hstack([y_val[idx][0:5], pred_val, predictions]).reshape((-1,)+pred_val.shape[1:]))],os.path.join(modelpath,f"multiple_05.png-{modality}"))

        
        models[modality] = model

        # loss, acc = models[modality].evaluate(x_test_noisy[idx], x_test[idx], verbose=2)
        
        # print(f"Test Loss {loss}\nTest Acc {acc}")
        
        # loss, acc = models[modality].evaluate(x_val_noisy[idx], x_val[idx], verbose=2)

        # print(f"Validation Loss {loss}\n Validation Acc {acc}")
        
    
    

    #test_result = t2_model.evaluate(x_test,y_val, verbose = 1)
    #print("Test accuracy :\t", round (test_result[1], 4))

   

    # predictions = t2_model.predict(x_test_noisy[1])
    # display([x_test_noisy[1], predictions])

    # predictions = t2_model.predict(x_val[1])
    # display([x_val[1], predictions])


    # predictions = t2_model.predict(x_val_noisy[1])
    # display([x_val_noisy[1], predictions])

    # predictions = t2_model.predict(x_val_noisy[1])
    # display([x_val_noisy[1], predictions])
    print("\n"+f"Job completed {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}".center(50, '*')+"\n")
    # keras.backend.clear_session()
    return #pat_df, x_train, x_test, x_val, x_train_noisy, x_test_noisy


# df, x_train, x_test, x_val, x_train_noisy, x_test_noisy = main()

if __name__ == '__main__':
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}") 
    print(f"SLURM_JOB_NAME - {os.environ['SLURM_JOB_NAME']}")
    print(f"SLURM_JOB_ID - {os.environ['SLURM_JOB_ID']}")
    print(f"tf version {tf.__version__}")
    
    if len(sys.argv)>1:
        print(sys.argv)
        kwargs={kw[0]:kw[1] for kw in [ar.split('=') for ar in sys.argv if ar.find('=')>0]}
        print(f"kwargs = {kwargs}")
        # main(**kwargs)
        main()
    else: main()

