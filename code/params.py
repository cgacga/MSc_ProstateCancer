import os, copy, tensorflow, time, numpy


class modality(dict):
    def __init__(self, *args, **kwargs):
        super(modality, self).__init__(*args, **kwargs)
        for key,value in self.items():
            setattr(modality, key, value)
    
    def mrkdown():
        c_dict = {v:m for v, m in vars(modality).items() if not (v.startswith('_')  or callable(m))}
        return to_json(c_dict)

class parameters(object):
    data_path: str
    tags = {}
    lst = {}
    _g = None
    
    def set_global(
                data_path : str,
                
                job_name : str = None,
                
                backbone_name : str = "vgg16", #vgg19
                classes : int = 1,
                activation : str = "sigmoid",
                encoder_weights : str = None, #"imagenet"
                encoder_freeze : bool = False,
                decoder_block_type : str = "upsampling",
                autoencoder_epocs : int = 100,
                autoencoder_batchsize : int = 2,
                autoencoder_learning_rate : float = 1e-4,
                classifier_train_learning_rate : float = 1e-5,
                classifier_test_learning_rate : float = 1e-5,
                no_adjacent : bool = False,
                minmax_shape_reduction : tuple = [5,15],
                minmax_augmentation_percentage : tuple = [10,15],
                mask_vs_rotation_percentage : int = 50,
                classifier_freeze_encoder : bool = False,
                classifier_multi_dense : bool = False,
                classifier_train_batchsize : int = 16,#16,
                classifier_train_epochs : int = 100,
                classifier_test_batchsize : int = 16,
                self_superviced = True,
                batchnorm = True,
                dropout = None,
                bootpercentage = 1,
                merge_method = "avg"):
        self = parameters()
        #Global parameters
        self.__class__.data_path = data_path
        #Common Parameters
        self.job_name = job_name
        #Model parameters
        self.modality_name = None
        self.model_name = None
        self.reshape_dim = None
        self.image_shape = tuple
        #Unet model Parameters
        #https://github.com/ZFTurbo/segmentation_models_3D
        #https://segmentation-models.readthedocs.io/en/latest/api.html#unet
        self.backbone_name = backbone_name
        self.encoder_weights = encoder_weights
        self.encoder_freeze = encoder_freeze
        self.decoder_block_type = decoder_block_type
        self.activation = activation
        self.classes = classes 
        #Training parameters

        self.modeltype = str
        self.self_superviced = self_superviced

        self.autoencoder_learning_rate = autoencoder_learning_rate
        self.autoencoder_epocs = autoencoder_epocs
        self.autoencoder_batchsize = autoencoder_batchsize
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.merge_method = merge_method

        self.classifier_freeze_encoder = classifier_freeze_encoder
        self.classifier_multi_dense = classifier_multi_dense
        self.classifier_train_batchsize = classifier_train_batchsize
        self.classifier_train_epochs = classifier_train_epochs
        self.classifier_train_learning_rate = classifier_train_learning_rate

        self.classifier_test_learning_rate = classifier_test_learning_rate
        self.classifier_test_batchsize = classifier_test_batchsize
        self.Nbootstrap_steps = 100
        self.bootpercentage = bootpercentage

        #Data augmentation parameters
        self.no_adjacent = no_adjacent
        self.minmax_shape_reduction = minmax_shape_reduction
        self.minmax_augmentation_percentage = minmax_augmentation_percentage
        self.mask_vs_rotation_percentage = mask_vs_rotation_percentage
        

        #Other parameters
        gpus = tensorflow.config.list_physical_devices('GPU')
        self.gpus = gpus if gpus else None
        self.n_gpus = len(self.gpus) if self.gpus  else 0
        self.dtime = time.strftime("%y%m%d_%H%M%S", time.localtime())
        self.ctime = time.ctime()
        self.skip_modality = False
        self.merged = False
        self.merged_modalities = tuple
        self.decode_try_upsample_first = bool
        self.encode_try_maxpool_first = bool
        self.same_shape = bool
        self.tag_idx = int
        self.encoder_method = str
        self.decoder_method = str
        self.decoder_filters = tuple
        self.center_filter = int
        self.tensorboard_num_predictimages = 5
        self.tensorboard_img_epoch = 50
        self.training_logs = {}
        self.__class__._g = self
    
    def add_modality(modality_name: str, reshape_dim :tuple or None, **kwargs):
        _g = copy.deepcopy(parameters._g)
        _g.modality_name = modality_name
        _g.model_name = f"{_g.modality_name}_{_g.job_name}_{_g.dtime}"
        _g.reshape_dim = reshape_dim
        for key, value in kwargs.items():
            if key in _g.__dict__:
              _g.__dict__[key] = value
            else:
                raise KeyError(f"{key} not a valid key. Check the spelling, valid keynames are: {*_g.add_modality.__annotations__.keys(),*_g.set_global.__annotations__.keys()}")
        
        missing = [key for key, x in _g.__dict__.items() if x == None and key != 'reshape_dim' and key != 'gpus' and key != 'encoder_weights' and key != "dropout" and key != "job_name"]
        if missing:
            raise KeyError(f"Missing value for: {missing}")
        else:
            root_path = f"{_g.job_name+'_' if _g.job_name else ''}selfsupervised_{_g.self_superviced}/"
                        
            C_name = f"/CTRlr{_g.classifier_train_learning_rate}_CTRfe{_g.classifier_freeze_encoder}_CTRmd{_g.classifier_multi_dense}_CTRbs{_g.classifier_train_batchsize}_CTRe{_g.classifier_train_epochs}/"
              
            C_eval_name = f"/CElr{_g.classifier_test_learning_rate}_CEbs{_g.classifier_test_batchsize}_CEbp{_g.bootpercentage}"

            AE_name = []
            AE_name.append(f"AEbn{_g.backbone_name}_AEw{_g.encoder_weights}_AElr{_g.autoencoder_learning_rate}_AEbn{_g.batchnorm}_AEd{_g.dropout}_AEsr{'-'.join(map(str, _g.minmax_shape_reduction))}_AEap{'-'.join(map(str, _g.minmax_augmentation_percentage))}_AEmvrp{_g.mask_vs_rotation_percentage}")
            AE_name.append(f"{modality_name}_AEe{_g.autoencoder_epocs}_AEbs{_g.autoencoder_batchsize}")
          
            if _g.merged:
              m_suffix = f"_AEem{_g.encoder_method}_AEdm{_g.decoder_method}_AEcf{_g.center_filter}_AEetmf{_g.encode_try_maxpool_first}_AEdtuf{_g.decode_try_upsample_first}_AEdf{_g.decoder_filters[0]}-{_g.decoder_filters[-1]}_AEmm{_g.merge_method}"
              AE_name[-1] = AE_name[-1]+m_suffix
            
            
            _g.AE_path = os.path.abspath(f'../models/{root_path+"/".join(AE_name)}/encoder/')

            _g.C_path = os.path.abspath(f'../models/{root_path+"/".join(AE_name)+C_name}/classifier/')
            _g.tensorboard_path = os.path.abspath(f'../tensorboard/{root_path+"/".join(AE_name)+C_name+C_eval_name}')
            _g.job_name = root_path+"/".join(AE_name)+C_name+C_eval_name

            parameters.lst[modality_name] = _g.__dict__
            if not _g.merged:
              parameters.tags[modality_name] = _g.reshape_dim

    def set_current(modality_name):
        try:
          modality(parameters.lst[modality_name])
        except KeyError:
          for key in parameters.lst.keys():
            if (key.find(modality_name) != -1):
              modality(parameters.lst[key])


    
    def insert_param(modality_name, key, value):
        parameters.lst[modality_name][key] = value

    def join_modalities(modality_names: list, encoder_method = "maxpool",decoder_method = "upsample", decoder_filters = (256, 128, 64, 32, 16), center_filter = 512, decode_try_upsample_first = True, encode_try_maxpool_first = True,merge_method="avg",**kwargs):
        up = ["upsample","transpose","padd"]
        down = ["maxpool","avgpool","reshape", "crop"]
        m_method = ["concat","avg","add","max","multiply"]
        if len(decoder_filters) != 5:
          raise ValueError("decoder_filters must be a list of 5 elements")
        if encoder_method  not in (up+down):
          raise ValueError(f"encoder_method: {encoder_method} must be one of {up+down}")
        if decoder_method  not in (up+down):
          raise ValueError(f"decoder_method: {decoder_method} must be one of {up+down}")
        if merge_method not in m_method:
          raise ValueError(f"Merge method: {merge_method} not supported, must be of {m_method}")
        modality_name = f"Merged_{'-'.join(modality_names)}"
        reshape_dim = tuple([parameters.tags[modality_name] for modality_name in modality_names])

        parameters.add_modality(modality_name, reshape_dim, merged_modalities = modality_names, encoder_method=encoder_method, decoder_method=decoder_method, decoder_filters=decoder_filters, center_filter=center_filter, merged=True, decode_try_upsample_first = decode_try_upsample_first, encode_try_maxpool_first = encode_try_maxpool_first,same_shape=False,merge_method=merge_method,**kwargs)

def to_json(o, level=0):
  indent = 4
  space = " "
  newline = "\n"

  ret = ""
  if isinstance(o, dict):
    ret += "{" + newline
    comma = ""
    for k, v in o.items():
      ret += comma
      comma = ",\n"
      ret += space * indent * (level + 1)
      ret += '"' + str(k) + '":' + space
      ret += to_json(v, level + 1)

    ret += newline + space * indent * level + "}"
  elif isinstance(o, str):
    ret += '"' + o + '"'
  elif isinstance(o, list):
    ret += "[" + ",".join([to_json(e, level + 1) for e in o]) + "]"
  elif isinstance(o, tuple):
    ret += "[" + ",".join(to_json(e, level + 1) for e in o) + "]"
  elif isinstance(o, bool):
    ret += "true" if o else "false"
  elif isinstance(o, int):
    ret += str(o)
  elif isinstance(o, float):
    ret += '%.7g' % o
  elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.integer):
    ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
  elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.inexact):
    ret += "[" + ','.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
  elif o is None:
    ret += 'null'
  elif isinstance(o, tensorflow.keras.optimizers.Optimizer):
    ret += o.get_config()["name"]
  elif isinstance(o, tensorflow.distribute.Strategy):
    ret += str(type(o))
  else:
    raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
  return "".join("\t" + line for line in ret.splitlines(True))