import numpy as np
import math
import cv2
from skimage import transform as stf

def transform(data, center, output_size, scale, rotation):
    scale_ratio = float(output_size)/scale
    rot = float(rotation)*np.pi/180.0
    #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = stf.SimilarityTransform(scale=scale_ratio)
    cx = center[0]*scale_ratio
    cy = center[1]*scale_ratio
    t2 = stf.SimilarityTransform(translation=(-1*cx, -1*cy))
    t3 = stf.SimilarityTransform(rotation=rot)
    t4 = stf.SimilarityTransform(translation=(output_size/2, output_size/2))
    t = t1+t2+t3+t4
    trans = t.params[0:2]
    #print('M', scale, rotation, trans)
    cropped = cv2.warpAffine(data,trans,(output_size, output_size), borderValue = 0.0)
    return cropped, trans

def transform2(data, label, output_size, bbox=None, dataset='ibug'):
    if bbox is None:
      record = np.zeros((4,), dtype=np.float32)
      for b in xrange(label.shape[0]):
        ind_gt = label[b]
        if b==0:
          record[0:2] = ind_gt
          record[2:4] = ind_gt
        else:
          record[0:2] = np.minimum(record[0:2], ind_gt)
          record[2:4] = np.maximum(record[2:4], ind_gt)
      if dataset=='ibug':
        record[1] = 0 if record[1]<36 else record[1]-36   # ibug
      elif dataset=='cofw_testset':
        record[1] = 0 if record[1]<45 else record[1]-45   # cofw_testset
      elif dataset=='300W':
        record[1] = 0 if record[1]<40 else record[1]-40   # 300W
      else:
        record[1] = 0 if record[1]<30 else record[1]-30   # AFLW2000-3D
      bbox = record
    trans = estimate_trans_bbox(bbox, output_size, s = 1.2)
    #print('M', scale, rotation, trans)
    cropped = cv2.warpAffine(data,trans,(output_size, output_size), borderValue = 0.0)
    # cv2.rectangle(data, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    # cv2.imshow("detection result", data)
    # cv2.waitKey(0)
    return cropped, trans

def transform_pt(pt, trans):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(trans, new_pt)
    #print('new_pt', new_pt.shape, new_pt)
    return new_pt[:2]

def gaussian(img, pt, sigma):
    # Draw a 2D gaussian
    assert(sigma>=0)
    if sigma==0:
      img[pt[1], pt[0]] = 1.0
      return True
    #assert pt[0]<=img.shape[1]
    #assert pt[1]<=img.shape[0]

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        #print('gaussian error')
        return False
        #return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 20
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) * 20  # multiply by 20

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return True
    #return img

def estimate_trans_bbox(face, input_size, s = 2.0):
  w = face[2] - face[0]
  h = face[3] - face[1]
  wc = int( (face[2]+face[0])/2 )
  hc = int( (face[3]+face[1])/2 )
  im_size = max(w, h)
  #size = int(im_size*1.2)
  scale = input_size/(max(w,h)*s)
  M = [ 
        [scale, 0, input_size/2-wc*scale],
        [0, scale, input_size/2-hc*scale],
      ]
  M = np.array(M)
  return M


def preprocess(data, label, output_size):
  M = None
  image_size = [data.shape[1], data.shape[0]]

  if label.shape[0]==68:
    landmark = np.zeros((5,2), dtype=np.float32)
    landmark[0,:] = (label[36,:]+label[39,:])/2   #left eye
    landmark[1,:] = (label[42,:]+label[45,:])/2   #right eye
    landmark[2,:] = label[30,:]                   #nose
    landmark[3,:] = label[48,:]                   #left mouth
    landmark[4,:] = label[54,:]                   #right mouth
  elif label.shape[0]==5:
    landmark = np.zeros((5,2), dtype=np.float32)
    landmark[0,:] = label[0,:]                   #left eye
    landmark[1,:] = label[1,:]                   #right eye
    landmark[2,:] = label[2,:]                   #nose
    landmark[3,:] = label[3,:]                   #left mouth
    landmark[4,:] = label[4,:]                   #right mouth
  # for i in range(5):
  #   cv2.circle(data, (landmark[i][0], landmark[i][1]), 1, (0, 0, 255), 2)
  # cv2.imshow("landmark", data)
  # cv2.waitKey(0)

  if landmark is not None:
    assert len(image_size)==2
    src = np.array([
      [38.2946, 41.6963],
      [73.5318, 41.5014],
      [56.0252, 61.7366],
      [41.5493, 82.3655],
      [70.7299, 82.2041] ], dtype=np.float32 )
    if output_size==384:
      src = src * 2 + 80.0
    dst = landmark.astype(np.float32)
    # for i in range(5):
    #   cv2.circle(data, (src[i][0], src[i][1]), 1, (0, 0, 255), 2)
    # cv2.imshow("landmark", data)
    # cv2.waitKey(0)

    tform = stf.SimilarityTransform()
    tform.estimate(dst, src)
    trans = tform.params[0:2,:]
    warped = cv2.warpAffine(data, trans, (output_size,output_size), borderValue = 0.0)

    label_out = np.zeros(label.shape, dtype=np.float32)
    for i in xrange(label.shape[0]):
      label_out[i] = transform_pt(label[i], trans)
    # for i in range(label.shape[0]):
    #   cv2.circle(warped, (label_out[i][0], label_out[i][1]), 1, (0, 0, 255), 2)
    # cv2.imshow("label", warped)
    # cv2.waitKey(0)

    return warped, label_out, trans