import numpy as np
from easydict import EasyDict as edict

config = edict()

#default training/dataset config
config.num_classes = 68
config.record_img_size = 384
config.base_scale = 256
config.input_img_size = 128
config.output_label_size = 64
config.label_xfirst = False
config.losstype = 'heatmap'
config.net_coherent = False
config.multiplier = 1.0

config.gaussian = 1

# topology settings
topology = edict()

topology.hourglass = edict()
topology.hourglass.net_coherent = False
topology.hourglass.net_sat = 0
topology.hourglass.net_n = 3
topology.hourglass.net_dcn = 0
topology.hourglass.net_stacks = 2
topology.hourglass.net_block = 'cab'
topology.hourglass.net_binarize = False
topology.hourglass.losstype = 'heatmap'

topology.sat = edict()
topology.sat.net_coherent = False
topology.sat.net_sat = 1
topology.sat.net_n = 3
topology.sat.net_dcn = 0  #3
topology.sat.net_stacks = 2
topology.sat.net_block = 'cab'
topology.sat.net_binarize = False
topology.sat.losstype = 'heatmap'


# dataset settings
dataset = edict()

dataset.i2d = edict()
dataset.i2d.dataset = '2D'
dataset.i2d.landmark_type = '2d'
dataset.i2d.dataset_path = '/media/3T_disk/my_datasets/sdu_net/data_2d'
dataset.i2d.num_classes = 68
dataset.i2d.record_img_size = 384
dataset.i2d.base_scale = 256
dataset.i2d.input_img_size = 128
dataset.i2d.output_label_size = 64
dataset.i2d.label_xfirst = False
dataset.i2d.val_targets = ['ibug', 'cofw_testset']  #'300W'

dataset.i3d = edict()
dataset.i3d.dataset = '3D'
dataset.i3d.landmark_type = '3d'
dataset.i3d.dataset_path = '/media/3T_disk/my_datasets/sdu_net/data_3d'
dataset.i3d.num_classes = 68
dataset.i3d.record_img_size = 384
dataset.i3d.base_scale = 256
dataset.i3d.input_img_size = 128
dataset.i3d.output_label_size = 64
dataset.i3d.label_xfirst = False
dataset.i3d.val_targets = ['AFLW2000-3D']


# default settings
default = edict()

# default topology
default.topology = 'sat'
default.pretrained = ''
default.pretrained_epoch = 0
# default dataset
default.dataset = 'i3d'
default.frequent = 20
default.verbose = 200
default.kvstore = 'device'

default.prefix = 'model/sat'
default.end_epoch = 10000
default.lr = 0.00025
default.wd = 0.0
default.per_batch_size = 16
default.lr_epoch_step = '20,35,45'

def generate_config(_topology, _dataset):
    for k, v in topology[_topology].items():
      config[k] = v
      default[k] = v
    for k, v in dataset[_dataset].items():
      config[k] = v
      default[k] = v
    config.topology = _topology
    config.dataset = _dataset

