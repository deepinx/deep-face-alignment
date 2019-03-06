import argparse
import cv2
import sys
import numpy as np
import os
import mxnet as mx
import datetime
import img_helper
from mtcnn_detector import MtcnnDetector


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
    #model = mx.mod.Module(symbol=sym, context=ctx)
    model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    self.model = model
    mtcnn_path = os.path.join(os.path.dirname(__file__),  'mtcnn_model')
    self.det_threshold = [0.6,0.7,0.8]
    self.detector = MtcnnDetector(model_folder=mtcnn_path, ctx=mx.cpu(), num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
  
  def get(self, img):
    ret = self.detector.detect_face(img)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    bbox = bbox[:,0:4]
    points = points[:,:].reshape((-1,2,5))
    points = np.transpose(points, (0,2,1))
    M = np.zeros( (bbox.shape[0], 2, 3), dtype=np.float32)
    ret = np.zeros( (bbox.shape[0], 68, 2), dtype=np.float32)
    for i in range(bbox.shape[0]):
      M[i] = img_helper.estimate_trans_bbox(bbox[i,:], self.image_size[0], s = 2.0)
      rimg = cv2.warpAffine(img, M[i], self.image_size, borderValue = 0.0)
      img2 = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
      img2 = np.transpose(img2, (2,0,1)) #3*112*112, RGB
      input_blob = np.zeros( (1, 3, self.image_size[1], self.image_size[0]),dtype=np.uint8 )
      input_blob[0] = img2
      ta = datetime.datetime.now()
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      self.model.forward(db, is_train=False)
      alabel = self.model.get_outputs()[-1].asnumpy()[0]
      tb = datetime.datetime.now()
      print('module time cost', (tb-ta).total_seconds())
      # ret = np.zeros( (alabel.shape[0], 2), dtype=np.float32)
      for j in xrange(alabel.shape[0]):
        a = cv2.resize(alabel[j], (self.image_size[1], self.image_size[0]))
        ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
        #ret[i] = (ind[0], ind[1]) #h, w
        ret[i,j] = (ind[1], ind[0]) #w, h
    return ret, M

ctx_id = 0
img_path = './sample-images/t1.jpg'
img = cv2.imread(img_path)
#img = np.zeros( (128,128,3), dtype=np.uint8 )

handler = Handler('./model_3d/sdu', 0, ctx_id)
for _ in range(2):
  ta = datetime.datetime.now() 
  ret, M2 = handler.get(img)
  tb = datetime.datetime.now()
  print('get time cost', (tb-ta).total_seconds())
#visualize landmark
for i in range(ret.shape[0]): 
  landmark = ret[i]
  M = M2[i]
  IM = cv2.invertAffineTransform(M)
  for i in range(landmark.shape[0]):
    p = landmark[i]
    point = np.ones( (3,), dtype=np.float32)
    point[0:2] = p
    point = np.dot(IM, point)
    landmark[i] = point[0:2]

  for i in range(landmark.shape[0]):
    p = landmark[i]
    point = (int(p[0]), int(p[1]))
    cv2.circle(img, point, 1, (0, 255, 0), 2)
cv2.imshow('alignment results', img)
cv2.waitKey(0)

filename = './landmark_test.png'
print('writing', filename)
cv2.imwrite(filename, img)



