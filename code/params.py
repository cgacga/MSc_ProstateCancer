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
    #strategy = None
    _g = None
    
    def set_global(
                data_path : str,
                
                job_name : str = None,
                
                backbone_name : str = "vgg16", #resnet18
                classes : int = 1,
                activation : str = "sigmoid",
                encoder_weights : str = "imagenet",
                encoder_freeze : bool = True,
                decoder_block_type : str = "upsampling", #transpose
                epochs : int = 100,
                #batch_size_prgpu : int = 2,
                batch_size : int = 2,
                optimizer : tensorflow.optimizers = tensorflow.optimizers.Adam(),
                loss : str = "mse",
                metrics : list[str] = ["mse", "mae"],
                learning_rate : float = 1e-4,
                no_adjacent : bool = False,
                minmax_shape_reduction : tuple = [5,15],
                minmax_augmentation_percentage : tuple = [10,15],
                mask_vs_rotation_percentage : int = 50):
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
        self.epochs = epochs
        self.batch_size = batch_size
        #self.batch_size_prgpu = batch_size_prgpu
        #self.global_batch_size = batch_size_prgpu # int
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics 
        self.learning_rate = learning_rate 
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
        self.same_shape = bool
        self.tag_idx = int
        #self.outputs = int
        self.encoder_method = str
        self.decoder_method = str
        self.decoder_filters = tuple
        self.center_filter = int
        self.tensorboard_num_predictimages = 5
        self.tensorboard_img_epoch = 10
        self.training_logs = {}
        #self.strategy = None
        #strategy = tensorflow.distribute.MirroredStrategy() if self.n_gpus>1 else tensorflow.distribute.get_strategy()
        
        
        #self.global_batch_size = self.batch_size_prgpu*strategy.num_replicas_in_sync
        #self.__class__.strategy = strategy

        self.__class__._g = self
    
    def add_modality(modality_name: str, reshape_dim :tuple or None, **kwargs):
        _g = copy.deepcopy(parameters._g)
        #_g.strategy = parameters.strategy
        _g.modality_name = modality_name
        _g.model_name = f"{_g.modality_name}_{_g.job_name}_{_g.dtime}"
        _g.reshape_dim = reshape_dim
        #merged = False
        for key, value in kwargs.items():
            if key in _g.__dict__:
              #if key == "merged_modalities":
                #merged = True
              _g.__dict__[key] = value
            else:
                #raise KeyError(f"{key} not a valid key, must be part of {_g.__dict__.keys()}")
                raise KeyError(f"{key} not a valid key. Check the spelling, valid keynames are: {*_g.add_modality.__annotations__.keys(),*_g.set_global.__annotations__.keys()}")
        
        missing = [key for key, x in _g.__dict__.items() if x == None and key != 'reshape_dim' and key != 'gpus']
        if missing:
            raise KeyError(f"Missing value for: {missing}")
        else:
            _g.learning_rate = _g.learning_rate*[_g.n_gpus if _g.n_gpus>0 else 1][0]
            _g.optimizer.lr.assign(_g.learning_rate)
            _g.model_path = os.path.abspath(f"../models/{_g.job_name}/{_g.dtime}/{modality_name}/")+"/"
            #_g.tensorboard_path = os.path.abspath(f"../tb_logs/{_g.job_name}/{_g.dtime}")+"/"#/{modality_name}")+"/"
            _g.tensorboard_path = os.path.abspath(f"../tb/{_g.job_name}/{modality_name}_{_g.dtime}")+"/"#/{modality_name}")+"/"
            parameters.lst[modality_name] = _g.__dict__
            #if not merged:
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

    def join_modalities(modality_names: list, encoder_method = "maxpool",decoder_method = "upsample", decoder_filters = (256, 128, 64, 32, 16), center_filter = 1024):
        up = ["upsample","transpose","padd"]
        down = ["maxpool","avgpool","reshape", "crop"]
        if len(decoder_filters) != 5:
          raise ValueError("decoder_filters must be a list of 5 elements")
        if encoder_method  not in (up+down):
          raise ValueError(f"encoder_method: {encoder_method} must be one of {up+down}")
        if decoder_method  not in (up+down):
          raise ValueError(f"decoder_method: {decoder_method} must be one of {up+down}")
        #_g = copy.deepcopy(parameters._g)
        #modality_name = f"Merged_{'_'.join(modality_names)}_"
        modality_name = f"Merged_{'-'.join(modality_names)}"
        #_g.model_name = f"{_g.modality_name}_{_g.job_name}_{_g.dtime}"
        reshape_dim = tuple([parameters.tags[modality_name] for modality_name in modality_names])

        parameters.add_modality(modality_name, reshape_dim, merged_modalities = modality_names, encoder_method=encoder_method, decoder_method=decoder_method, decoder_filters=decoder_filters, center_filter=center_filter, merged=True)

        # for modality_name in modality_names:
        #     for key, value in parameters.lst[modality_name].items():
        #         if key in _g.__dict__:
        #             _g.__dict__[key] = value
        #         else:
        #             #raise KeyError(f"{key} not a valid key, must be part of {_g.__dict__.keys()}")
        #             raise KeyError(f"{key} not a valid key. Check the spelling, valid keynames are: {*_g.add_modality.__annotations__.keys(),*_g.set_global.__annotations__.keys()}")
        # missing = [key for key, x in _g.__dict__.items() if x == None and key != 'reshape_dim' and key != 'gpus']
        # if missing:
        #     raise KeyError(f"Missing value for: {missing}")
        # else:
        #     _g.learning_rate = _g.learning_rate*[_g.n_gpus if _g.n_gpus>0 else 1][0]
        #     _g.optimizer.lr.assign(_g.learning_rate)
        #     _g.model_path = os.path.abspath(f"../models/{_g.job_name}/{_g.dtime}/{_g.modality_name}/")+"/"
        #     _g.tensorboard_path = os.path.abspath(f"../tb_logs/{_g.job_name}/{_g.dtime}")+"/"#/{_g.modality_name}")+"/"
        #     parameters.lst[_g.modality_name] = _g.__dict__
        #     parameters.tags[_g.modality_name] = _g.reshape_dim




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
  # Tuples are interpreted as lists
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