from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
from data import FaceSegIter
import mxnet as mx
import mxnet.optimizer as optimizer
import numpy as np
import os
import sys
import math
import random
import cv2
from config import config, default, generate_config
from optimizer import ONadam
from metric import LossValueMetric, NMEMetric
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
import hourglass
import satnet
#import sym_fc
#from symbol import fc


args = None
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def init_weights(sym, data_shape_dict):
  #print('in hg')
  arg_name = sym.list_arguments()
  aux_name = sym.list_auxiliary_states()
  arg_shape, _, aux_shape = sym.infer_shape(**data_shape_dict)
  arg_shape_dict = dict(zip(arg_name, arg_shape))
  aux_shape_dict = dict(zip(aux_name, aux_shape))
  #print(aux_shape)
  #print(aux_params)
  #print(arg_shape_dict)
  arg_params = {}
  aux_params = {}
  for k,v in arg_shape_dict.iteritems():
    #print(k,v)
    if k.endswith('offset_weight') or k.endswith('offset_bias'):
      print('initializing',k)
      arg_params[k] = mx.nd.zeros(shape = v)
    elif k.startswith('fc6_'):
      if k.endswith('_weight'):
        print('initializing',k)
        arg_params[k] = mx.random.normal(0, 0.01, shape=v)
      elif k.endswith('_bias'):
        print('initializing',k)
        arg_params[k] = mx.nd.zeros(shape=v)
    elif k.find('upsampling')>=0:
      print('initializing upsampling_weight', k)
      arg_params[k] = mx.nd.zeros(shape=arg_shape_dict[k])
      init = mx.init.Initializer()
      init._init_bilinear(k, arg_params[k])
  return arg_params, aux_params

def val_test(sym, model, ctx, data_shape, global_step):
  results = []
  test_batch_size = 1
  all_layers = sym.get_internals()
  vsym = all_layers['heatmap_output']
  vmodel = mx.mod.Module(symbol=vsym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  vmodel.bind(data_shapes=[('data', (test_batch_size,)+data_shape)])
  arg_params, aux_params = model.get_params()
  vmodel.set_params(arg_params, aux_params)
  for target in config.val_targets:
      _file = os.path.join(config.dataset_path, '%s.rec'%target)
      if not os.path.exists(_file):
          continue
      val_iter = FaceSegIter(path_imgrec = _file,
        # batch_size = args.batch_size,
        batch_size = test_batch_size,
        aug_level = 0,
        args = args,
        )
      _metric = NMEMetric()
      val_metric = mx.metric.create(_metric)
      val_metric.reset()
      val_iter.reset()
      nme = []
      for i, eval_batch in enumerate(val_iter):
        #print(eval_batch.data[0].shape, eval_batch.label[0].shape)
        batch_data = mx.io.DataBatch(eval_batch.data)
        vmodel.forward(batch_data, is_train=False)
        # vmodel.update_metric(val_metric, eval_batch.label)
        pred_label = vmodel.get_outputs()[-1].asnumpy()
        label = eval_batch.label[0].asnumpy()
        _nme = _metric.cal_nme(label, pred_label)
        nme.append(_nme)
      # nme_value = val_metric.get_name_value()[0][1]
      nme_value = np.mean(nme)
      results.append(nme_value)
      print('[%d][%s]NME: %f'%(global_step[0], target, nme_value))
  return results


def main(args):
  _seed = 727
  random.seed(_seed)
  np.random.seed(_seed)
  mx.random.seed(_seed)
  ctx = []
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
  if len(cvd)>0:
    for i in xrange(len(cvd.split(','))):
      ctx.append(mx.gpu(i))
  if len(ctx)==0:
    ctx = [mx.cpu()]
    print('use cpu')
  else:
    print('gpu num:', len(ctx))
  #ctx = [mx.gpu(0)]
  args.ctx_num = len(ctx)

  args.batch_size = args.per_batch_size*args.ctx_num
  config.per_batch_size = args.per_batch_size

  print('Call with', args, config)
  train_iter = FaceSegIter(path_imgrec = os.path.join(config.dataset_path, 'train.rec'),
      batch_size = args.batch_size,
      per_batch_size = args.per_batch_size,
      aug_level = 1,
      exf = args.exf,
      args = args,
      )

  data_shape, data_size = train_iter.get_data_shape()
  #label_shape = train_iter.get_label_shape()
  sym = eval(config.network).get_symbol(num_classes=config.num_classes)
  if len(args.pretrained)==0:
      #data_shape_dict = {'data' : (args.per_batch_size,)+data_shape, 'softmax_label' : (args.per_batch_size,)+label_shape}
      data_shape_dict = train_iter.get_shape_dict()
      arg_params, aux_params = init_weights(sym, data_shape_dict)
  else:
      vec = args.pretrained.split(',')
      print('loading', vec)
      _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
      #sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)

  model = mx.mod.Module(
      context       = ctx,
      symbol        = sym,
      label_names   = train_iter.get_label_names(),
  )
  #lr = 1.0e-3
  #lr = 2.5e-4
  _rescale_grad = 1.0/args.ctx_num
  #_rescale_grad = 1.0/args.batch_size
  #lr = args.lr
  #opt = optimizer.Nadam(learning_rate=args.lr, wd=args.wd, rescale_grad=_rescale_grad, clip_gradient=5.0)
  if args.optimizer=='onadam':
    opt = ONadam(learning_rate=args.lr, wd=args.wd, rescale_grad=_rescale_grad, clip_gradient=5.0)
  elif args.optimizer=='nadam':
    opt = optimizer.Nadam(learning_rate=args.lr, rescale_grad=_rescale_grad)
  elif args.optimizer=='rmsprop':
    opt = optimizer.RMSProp(learning_rate=args.lr, rescale_grad=_rescale_grad)
  elif args.optimizer=='adam':
    opt = optimizer.Adam(learning_rate=args.lr, rescale_grad=_rescale_grad)
  else:
    opt = optimizer.SGD(learning_rate=args.lr, momentum=0.9, wd=args.wd, rescale_grad=_rescale_grad)
  initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
  _cb = mx.callback.Speedometer(args.batch_size, args.frequent)
  _metric = LossValueMetric()
  #_metric = NMEMetric()
  #_metric2 = AccMetric()
  #eval_metrics = [_metric, _metric2]
  eval_metrics = [_metric]
  lr_epoch_steps = [int(x) for x in args.lr_epoch_step.split(',')]
  print('lr-epoch-steps', lr_epoch_steps)
  
  global_step = [0]
  highest_acc = [1.0, 1.0]
  def _batch_callback(param):
    _cb(param)
    global_step[0]+=1
    mbatch = global_step[0]
    mepoch = mbatch*args.batch_size//data_size
    pre = mbatch*args.batch_size%data_size
    is_highest = False
    for _lr in lr_epoch_steps[0:-1]:
      if mepoch==_lr and pre<args.batch_size:
        opt.lr *= 0.2
        print('lr change to', opt.lr)
        break
    if mbatch%1000==0:
      print('lr:',opt.lr,'batch:',param.nbatch,'epoch:',param.epoch)
    if mbatch>0 and mbatch%args.verbose==0:
      acc_list = val_test(sym, model, ctx, data_shape, global_step)
      score = np.mean(acc_list) 
      if acc_list[0]<highest_acc[0]:  # ibug
        is_highest = True
        highest_acc[0] = acc_list[0]
      if score<highest_acc[1]:   # mean
        is_highest = True
        highest_acc[1] = score
      if args.ckpt==1 and is_highest==True:
        msave = mbatch//args.verbose
        print('saving', msave)
        arg, aux = model.get_params()
        mx.model.save_checkpoint(args.prefix, msave, model.symbol, arg, aux)
    if mepoch==lr_epoch_steps[-1]:
      if args.ckpt==1:
        acc_list = val_test(sym, model, ctx, data_shape, global_step)
        msave = mbatch//args.verbose
        print('saving', msave)
        arg, aux = model.get_params()
        mx.model.save_checkpoint(args.prefix, msave, model.symbol, arg, aux)
      sys.exit(0)

  train_iter = mx.io.PrefetchingIter(train_iter)

  model.fit(train_iter,
      begin_epoch        = 0,
      num_epoch          = 9999,
      #eval_data          = val_iter,
      eval_data          = None,
      eval_metric        = eval_metrics,
      kvstore            = 'device',
      optimizer          = opt,
      initializer        = initializer,
      arg_params         = arg_params,
      aux_params         = aux_params,
      allow_missing      = True,
      batch_end_callback = _batch_callback,
      epoch_end_callback = None,
      )

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train face alignment')
  # general
  parser.add_argument('--network', help='network name', default=default.network, type=str)
  parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
  args, rest = parser.parse_known_args()
  generate_config(args.network, args.dataset)
  parser.add_argument('--prefix', default=default.prefix, help='directory to save model.')
  parser.add_argument('--pretrained', default=default.pretrained, help='')
  parser.add_argument('--optimizer', default='nadam', help='')
  parser.add_argument('--lr', type=float, default=default.lr, help='')
  parser.add_argument('--wd', type=float, default=default.wd, help='')
  parser.add_argument('--per-batch-size', type=int, default=default.per_batch_size, help='')
  parser.add_argument('--lr-epoch-step', help='learning rate steps (in epoch)', default=default.lr_epoch_step, type=str)
  parser.add_argument('--ckpt', type=int, default=1, help='')
  parser.add_argument('--norm', type=int, default=0, help='')
  parser.add_argument('--exf', type=int, default=1, help='')
  parser.add_argument('--frequent', type=int, default=default.frequent, help='')
  parser.add_argument('--verbose', type=int, default=default.verbose, help='')
  args = parser.parse_args()
  main(args)

