import argparse
import cv2
import sys
import os
import numpy as np
import mxnet as mx
import datetime
import img_helper
from config import config
import matplotlib.pyplot as plt
from essh_detector import ESSHDetector
from metric import LossValueMetric, NMEMetric


class Handler:
  def __init__(self, prefix, epoch, ctx_id=0):
    print('loading',prefix, epoch)
    if ctx_id>=0:
      ctx = mx.gpu(ctx_id)
    else:
      ctx = mx.cpu()
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers['heatmap_output']
    image_size = (128, 128)
    self.image_size = image_size
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    self.model = model
    self.detector = ESSHDetector('./essh-model/essh', 0)

  def trans_dot(self, trans1, trans2):
    trans1 = np.vstack((trans1, [0,0,1]))
    trans2 = np.vstack((trans2, [0,0,1]))
    trans21 = np.dot(trans2, trans1)[0:2]
    return trans21

  def get_maxpos(self, img, det):
    img_size = np.asarray(img.shape)[0:2]
    # bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
    img_center = img_size / 2
    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
    # bindex = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
    bindex = np.argmin(offset_dist_squared) # some extra weight on the centering
    return bindex
  
  def get_landmark(self, img, label, dataset, use_essh):
    if use_essh:
      ret = self.detector.detect(img, threshold=0.4)
      if ret is None or ret.shape[0]==0:
        return None, None
      bindex = self.get_maxpos(img, ret)
      face = ret[bindex]
      bbox = face[0:4]
      points = face[5:15].reshape(5,2)
      # b = bbox
      # cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))
      # for p in landmark:
      #   cv2.circle(img, (int(p[0]), int(p[1])), 1, (0, 0, 255), 2)
      # cv2.imshow("detection result", img)
      # cv2.waitKey(0)
      # for i in range(bbox.shape[0]):
      rimg, label2, trans1 = img_helper.preprocess(img, points, img.shape[0])
      ret2 = self.detector.detect(rimg, threshold=0.4)
      if ret2 is None or ret2.shape[0]==0:
        return None, None
      bindex2 = self.get_maxpos(rimg, ret2)
      rimg, trans2 = img_helper.transform2(rimg, None, self.image_size[0], ret2[bindex2,0:4], dataset)
    else:
      rimg, label2, trans1 = img_helper.preprocess(img, label, img.shape[0])
      rimg, trans2 = img_helper.transform2(rimg, label2, self.image_size[0], None, dataset)
    trans = self.trans_dot(trans1, trans2)
    # cv2.imshow("rimg", rimg)
    # cv2.waitKey(0)
    # img2 = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
    img2 = np.transpose(rimg, (2,0,1)) #3*128*128, RGB
    input_blob = np.zeros( (1, 3, self.image_size[1], self.image_size[0]),dtype=np.uint8 )
    input_blob[0] = img2
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    alabel = self.model.get_outputs()[-1].asnumpy()[0]
    IM = cv2.invertAffineTransform(trans)
    landmark = np.zeros( (68, 2), dtype=np.float32)
    for i in xrange(alabel.shape[0]):
      a = cv2.resize(alabel[i], (self.image_size[1], self.image_size[0]))
      ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
      point = (ind[1], ind[0], 1.0) #w, h
      point = np.dot(IM, point)
      landmark[i] = point[0:2]
      npt = img_helper.transform_pt(label[i], trans)
      if config.landmark_type=='2d':
        npt = np.floor(npt)
      else:
        npt = np.round(npt)
      point = (npt[0], npt[1], 1.0)
      point = np.dot(IM, point)
      label[i] = point[0:2]
    return landmark, label
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='test nme on rec data')
  # general
  parser.add_argument('--dataset', default='ibug', help='test dataset name')
  parser.add_argument('--prefix', default='./models/model-hg2d3-cab/model', help='model prefix')
  parser.add_argument('--epoch', type=int, default=0, help='model epoch')
  parser.add_argument('--gpu', type=int, default=0, help='')
  parser.add_argument('--landmark-type', default='2d', help='')
  parser.add_argument('--image-size', type=int, default=128, help='')
  parser.add_argument('--use-essh', type=bool, default=False, help='')
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
  use_essh = args.use_essh
  image_size = (args.image_size, args.image_size)
  config.landmark_type = args.landmark_type
  config.input_img_size = image_size[0]
  config.use_essh = args.use_essh

  handler = Handler(prefix=prefix, epoch=epoch, ctx_id=ctx_id)
  idx_path = rec_path[0:-4]+".idx"
  imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')  
  seq = list(imgrec.keys)
  _metric = NMEMetric()
  nme = []
  miss = 0
  for img_idx in seq:
    if img_idx%10==0:
      print('processing %d' %img_idx)
    s = imgrec.read_idx(img_idx)
    header, img = mx.recordio.unpack(s)
    try:
      img = mx.image.imdecode(img).asnumpy()
    except:
      continue
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hlabel = np.array(header.label).reshape((68, 2))
    hlabel = hlabel[:,::-1] #convert to X/W first
    preds, label = handler.get_landmark(img, hlabel, dataset, use_essh)
    if preds is None:
      print('no face detected %d' %img_idx)
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      cv2.imwrite('sample-images/miss/%d.jpg'%img_idx, img)
      miss += 1
      continue
    # label = hlabel[np.newaxis, :, :]
    # pred_label = preds[np.newaxis, :, :]
    _nme = _metric.calculate_nme(label, preds)
    nme.append(_nme)

    # visualize landmark
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
  print('total miss num is %d' %miss)
  print('nme on %s is %.3f%%' %(dataset, np.mean(nme)*100))
