import mxnet as mx
import numpy as np
import sys, os
import cv2


# dataset = 'data_2d/train'
# dataset = 'data_2d/ibug'
# dataset = 'data_2d/300W'
# dataset = 'data_2d/cofw_testset'
# dataset = 'data_3d/train'
dataset = 'data_3d/AFLW2000-3D'
source_dir = '/media/3T_disk/my_datasets/sdu_net/'
output_dir = '/media/3T_disk/my_datasets/sdu_net/%s'%dataset

print('starting to convert %s' %dataset)

source_idx = os.path.join(source_dir, '%s.idx'%dataset)
source_rec = os.path.join(source_dir, '%s.rec'%dataset)
imgrec = mx.recordio.MXIndexedRecordIO(source_idx, source_rec, 'r')  
seq = list(imgrec.keys)
widx = 0
for img_idx in seq:
  if img_idx%1000==0:
    print('processing %s %d' %(dataset,img_idx))
  s = imgrec.read_idx(img_idx)
  header, img = mx.recordio.unpack(s)
  try:
    image = mx.image.imdecode(img).asnumpy()
  except:
    continue
  img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  hlabel = np.array(header.label).reshape((68, 2))
  hlabel = hlabel[:,::-1] #convert to X/W first

  for i in range(hlabel.shape[0]):
    p = hlabel[i]
    point = (int(p[0]), int(p[1]))
    cv2.circle(img, point, 1, (0, 255, 0), 2)
  
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  filename = '%s/%d.jpg'%(output_dir,img_idx)
  # print('writing', filename)
  cv2.imwrite(filename, img)



