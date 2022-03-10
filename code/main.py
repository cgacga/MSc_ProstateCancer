#%%
### Importing libraries ###
import os
import time
import sys
import numpy as np
import tensorflow as tf
# from tensorflow.keras import backend as K



from data_augmentation import *
from preprocess import *
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
import random
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
    #tags = {"ADC": None,"t2tsetra": (32,320,320)} 
    tags = {"t2tsetra": (32,320,320)} 

    pat_slices, pat_df = preprocess(data_path,tags,False)
    x_train, x_test, x_val, x_train_noisy, x_test_noisy, x_val_noisy = data_augmentation(pat_slices, pat_df)
    # x_train, _, _, x_train_noisy, _, _ = data_augmentation(pat_slices, pat_df)
    # del pat_slices

    
    def sizeof_fmt(num, suffix='B'):
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)

    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                            key= lambda x: -x[1])[:30]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


    


    models = {}
    for modality in tags.keys():
        # modelpath = f"../models/{modality}/{os.environ['SLURM_JOB_NAME']}/{os.environ['SLURM_JOB_ID']}-{os.environ['SLURM_JOB_NAME']}"
        modelpath = f"../models/{modality}/{os.environ['SLURM_JOB_NAME']}"

        shape,idx = pat_df[["dim","tag_idx"]][pat_df.tag.str.contains(modality, case=False)].values[0]
        # train_data = tf.data.Dataset.from_tensor_slices((x_train_noisy[idx], x_train[idx]))
        # val_data = tf.data.Dataset.from_tensor_slices((x_test_noisy[idx], x_test[idx]))
        # models[modality] = model_building(shape, modelpath, train_data, val_data)
        print(x_train_noisy[idx].shape)
        print(x_train[idx].shape)
        print(x_test_noisy[idx].shape)
        print(x_test[idx].shape)
        #https://stackoverflow.com/questions/52724022/model-fits-on-a-single-gpu-but-script-crashes-when-trying-to-fit-on-multiple-gpu
        print(x_val_noisy[idx].shape)
        print(x_val[idx].shape)
        model = model_building(shape, modelpath, x_train_noisy[idx],x_train[idx], x_test_noisy[idx], x_test[idx])
 

        pred_test = tf.expand_dims(x_test_noisy[idx][0],axis=0)
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
        img_pltsave([x_test[idx][0], x_test_noisy[idx][0], predictions],os.path.join(modelpath,"test_set.png"))

        pred_val = tf.expand_dims(x_val_noisy[idx][0],axis=0)
        if len(gpus)>1:
            with strategy.scope():
                predictions = model.predict(pred_val)
        else: predictions = model.predict(pred_val)
        img_pltsave([x_val[idx][0], x_val_noisy[idx][0], predictions],os.path.join(modelpath,"validation_set.png"))

        models[modality] = model

        # loss, acc = models[modality].evaluate(x_test_noisy[idx], x_test[idx], verbose=2)
        
        # print(f"Test Loss {loss}\nTest Acc {acc}")
        
        # loss, acc = models[modality].evaluate(x_val_noisy[idx], x_val[idx], verbose=2)

        # print(f"Validation Loss {loss}\n Validation Acc {acc}")
        
    
    

    #test_result = t2_model.evaluate(x_test,y_test, verbose = 1)
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

