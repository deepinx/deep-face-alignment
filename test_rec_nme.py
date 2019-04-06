import argparse
import cv2
import sys
import numpy as np
import os
import mxnet as mx
import datetime
import img_helper
from config import config
import matplotlib.pyplot as plt
from data import FaceSegIter
from metric import LossValueMetric, NMEMetric


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='test nme on rec data')
  # general
  parser.add_argument('--dataset', default='ibug', help='test dataset name')
  parser.add_argument('--prefix', default='./models/model-hg2d3-cab/model', help='model prefix')
  parser.add_argument('--epoch', type=int, default=0, help='model epoch')
  parser.add_argument('--gpu', type=int, default=0, help='')
  parser.add_argument('--landmark-type', default='2d', help='')
  parser.add_argument('--image-size', type=int, default=128, help='')
  args = parser.parse_args()

  if args.dataset=='ibug':
    rec_path = '/media/3T_disk/my_datasets/sdu_net/data_2d/ibug.rec'
  elif args.dataset=='cofw_testset':
    rec_path = '/media/3T_disk/my_datasets/sdu_net/data_2d/cofw_testset.rec'
  elif args.dataset=='300W':
    rec_path = '/media/3T_disk/my_datasets/sdu_net/data_2d/300W.rec'
  else:
    rec_path = '/media/3T_disk/my_datasets/sdu_net/data_3d/AFLW2000-3D.rec'
  dataset = args.dataset
  ctx_id = args.gpu
  prefix = args.prefix
  epoch = args.epoch
  image_size = (args.image_size, args.image_size)
  config.landmark_type = args.landmark_type
  config.input_img_size = image_size[0]

  if ctx_id>=0:
    ctx = mx.gpu(ctx_id)
  else:
    ctx = mx.cpu()
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers['heatmap_output']
  #model = mx.mod.Module(symbol=sym, context=ctx, data_names=['data'], label_names=['softmax_label'])
  model = mx.mod.Module(symbol=sym, context=ctx, data_names=['data'], label_names=None)
  #model = mx.mod.Module(symbol=sym, context=ctx)
  model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)

  val_iter = FaceSegIter(path_imgrec = rec_path,
    batch_size = 1,
    aug_level = 0,
    )
  _metric = NMEMetric()
  #val_metric = mx.metric.create(_metric)
  #val_metric.reset()
  #val_iter.reset()
  nme = []
  for i, eval_batch in enumerate(val_iter):
    if i%10==0:
      print('processing', i)
    # print(eval_batch.data[0].shape, eval_batch.label[0].shape)
    
    batch_data = mx.io.DataBatch(eval_batch.data)
    model.forward(batch_data, is_train=False)
    #model.update_metric(val_metric, eval_batch.label, True)
    pred_label = model.get_outputs()[-1].asnumpy()
    gt_label = eval_batch.label[0].asnumpy()
    _nme = _metric.cal_nme(gt_label, pred_label)
    nme.append(_nme)

    # visualize landmark
    # preds = np.zeros( (68, 2), dtype=np.float32)
    # label = np.zeros( (68, 2), dtype=np.float32)
    # for i in xrange(pred_label.shape[1]):
    #   a = cv2.resize(pred_label[0][i], image_size)
    #   b = cv2.resize(gt_label[0][i], image_size)
    #   ind_a = np.unravel_index(np.argmax(a, axis=None), a.shape)
    #   ind_b = np.unravel_index(np.argmax(b, axis=None), b.shape)
    #   preds[i] = (ind_a[1], ind_a[0]) #w, h
    #   label[i] = (ind_b[1], ind_b[0]) #w, h
    # _nme = _metric.calculate_nme(label, preds)
    # nme.append(_nme)
    # data = np.transpose(eval_batch.data[0].asnumpy()[0], (1,2,0))
    # img = data.astype(np.uint8)
    # plt.imshow(img)
    # plt.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    # plt.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    # plt.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    # plt.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    # plt.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    # plt.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    # plt.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    # plt.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    # plt.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5) 
    # plt.plot(label[0:17,0],label[0:17,1],marker='o',markersize=1,linestyle='-',color='y',lw=0.5)
    # plt.plot(label[17:22,0],label[17:22,1],marker='o',markersize=1,linestyle='-',color='y',lw=0.5)
    # plt.plot(label[22:27,0],label[22:27,1],marker='o',markersize=1,linestyle='-',color='y',lw=0.5)
    # plt.plot(label[27:31,0],label[27:31,1],marker='o',markersize=1,linestyle='-',color='y',lw=0.5)
    # plt.plot(label[31:36,0],label[31:36,1],marker='o',markersize=1,linestyle='-',color='y',lw=0.5)
    # plt.plot(label[36:42,0],label[36:42,1],marker='o',markersize=1,linestyle='-',color='y',lw=0.5)
    # plt.plot(label[42:48,0],label[42:48,1],marker='o',markersize=1,linestyle='-',color='y',lw=0.5)
    # plt.plot(label[48:60,0],label[48:60,1],marker='o',markersize=1,linestyle='-',color='y',lw=0.5)
    # plt.plot(label[60:68,0],label[60:68,1],marker='o',markersize=1,linestyle='-',color='y',lw=0.5) 
    # plt.axis('off')
    # plt.show()
  print('total NME is %.3f%%' %(np.mean(nme)*100))

