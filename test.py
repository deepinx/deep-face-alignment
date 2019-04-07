import argparse
import cv2
import sys
import numpy as np
import os
import mxnet as mx
import datetime
import img_helper
import matplotlib.pyplot as plt
from essh_detector import ESSHDetector

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
  
  def get_landmark(self, img):
    ret = self.detector.detect(img, threshold=0.4)
    if ret is None or ret.shape[0]==0:
      return None
    bbox = ret[:,0:4]
    points = ret[:, 5:15].reshape(-1,5,2)
    landmark_list = np.zeros( (bbox.shape[0], 68, 2), dtype=np.float32)
    for i in range(bbox.shape[0]):
      rimg, label, trans1 = img_helper.preprocess(img, points[i], 384)
      ret2 = self.detector.detect(rimg, threshold=0.4)
      rimg, trans2 = img_helper.transform2(rimg, None, self.image_size[0], ret2[0,0:4])
      trans = self.trans_dot(trans1, trans2)
      # cv2.imshow("rimg", rimg)
      # cv2.waitKey(0)
      img2 = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
      img2 = np.transpose(img2, (2,0,1)) #3*128*128, RGB
      input_blob = np.zeros( (1, 3, self.image_size[1], self.image_size[0]),dtype=np.uint8 )
      input_blob[0] = img2
      ta = datetime.datetime.now()
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      self.model.forward(db, is_train=False)
      alabel = self.model.get_outputs()[-1].asnumpy()[0]
      tb = datetime.datetime.now()
      print('module time cost', (tb-ta).total_seconds())
      IM = cv2.invertAffineTransform(trans)
      for j in xrange(alabel.shape[0]):
        a = cv2.resize(alabel[j], (self.image_size[1], self.image_size[0]))
        ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
        # landmark_list[i] = (ind[0], ind[1]) #h, w
        # landmark_list[i,j] = (ind[1], ind[0]) #w, h
        point = (ind[1], ind[0], 1.0)
        point = np.dot(IM, point)
        landmark_list[i,j] = point[0:2]
    return landmark_list


if __name__ == '__main__':
  img_path = './sample-images/t2.jpg'
  prefix = './models/model-hg2d3-cab-3d/model'
  img = cv2.imread(img_path)
  handler = Handler(prefix=prefix, epoch=0, ctx_id=0)
  for _ in range(2):
    ta = datetime.datetime.now() 
    landmark_list = handler.get_landmark(img)
    tb = datetime.datetime.now()
    print('get time cost', (tb-ta).total_seconds())

  # visualize landmark
  img2 = plt.imread(img_path)
  plt.imshow(img2)
  for i in range(landmark_list.shape[0]): 
    landmark = landmark_list[i]
    preds = landmark
    plt.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    plt.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    plt.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    plt.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    plt.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    plt.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    plt.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    plt.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5)
    plt.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=0.5) 
  plt.axis('off')
  plt.show()



