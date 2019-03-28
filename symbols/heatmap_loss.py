from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mxnet as mx
import numpy as np
from config import config



class SymCoherent:
  def __init__(self, per_batch_size):
    self.per_batch_size = per_batch_size
    self.flip_order = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 
        26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33, 32, 31, 
        45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51, 50, 49, 48, 
        59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65]

  def get(self, data):
    #data.shape[0]==per_batch_size
    b = self.per_batch_size//2
    ux = mx.sym.slice_axis(data, axis=0, begin=0, end=b)
    dx = mx.sym.slice_axis(data, axis=0, begin=b, end=b*2)
    ux = mx.sym.flip(ux, axis=3)
    #ux = mx.sym.take(ux, indices = self.flip_order, axis=0)
    ux_list = []
    for o in self.flip_order:
      _ux = mx.sym.slice_axis(ux, axis=1, begin=o, end=o+1)
      ux_list.append(_ux)
    ux = mx.sym.concat(*ux_list, dim=1)
    return ux, dx

def l2_loss(x, y):
  loss = x-y
  # loss = mx.symbol.smooth_l1(loss, scalar=1.0)
  loss = loss*loss
  loss = mx.symbol.mean(loss)
  return loss

def ce_loss(x, y):
  #loss = mx.sym.SoftmaxOutput(data = x, label = y, normalization='valid', multi_output=True)
  x_max = mx.sym.max(x, axis=[2,3], keepdims=True)
  x = mx.sym.broadcast_minus(x, x_max)
  #x = mx.sym.L2Normalization(x, mode='instance')
  body = mx.sym.exp(x)
  sums = mx.sym.sum(body, axis=[2,3], keepdims=True)
  body = mx.sym.broadcast_div(body, sums)
  loss = mx.sym.log(body)
  loss = loss*y*(-1.0)
  #loss = mx.symbol.mean(loss, axis=[1,2,3])
  loss = mx.symbol.mean(loss)
  return loss

# def get_symbol(num_classes):
#     m = config.multiplier
#     sFilters = max(int(64*m), 32)
#     mFilters = max(int(128*m), 32)
#     nFilters = int(256*m)

#     nModules = 1
#     nStacks = config.net_stacks
#     binarize = config.net_binarize
#     input_size = config.input_img_size
#     label_size = config.output_label_size
#     use_coherent = config.net_coherent
#     use_SAT = config.net_sat
#     N = config.net_n
#     DCN = config.net_dcn
#     per_batch_size = config.per_batch_size
#     print('binarize', binarize)
#     print('use_coherent', use_coherent)
#     print('use_SAT', use_SAT)
#     print('use_N', N)
#     print('use_DCN', DCN)
#     print('per_batch_size', per_batch_size)
#     #assert(label_size==64 or label_size==32)
#     #assert(input_size==128 or input_size==256)
#     coherentor = SymCoherent(per_batch_size)
#     D = input_size // label_size
#     print(input_size, label_size, D)
#     data = mx.sym.Variable(name='data')
#     data = data-127.5
#     data = data*0.0078125
#     gt_label = mx.symbol.Variable(name='softmax_label')
#     losses = []
#     closses = []
#     ref_label = gt_label
#     if D==4:
#       body = Conv(data=data, num_filter=sFilters, kernel=(7, 7), stride=(2,2), pad=(3, 3),
#                               no_bias=True, name="conv0", workspace=workspace)
#     else:
#       body = Conv(data=data, num_filter=sFilters, kernel=(3, 3), stride=(1,1), pad=(1, 1),
#                               no_bias=True, name="conv0", workspace=workspace)
#     body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
#     body = Act(data=body, act_type='relu', name='relu0')

#     dcn = False
#     body = conv_block(body, mFilters, (1,1), sFilters==mFilters, 'res0', False, dcn, 1)

#     body = mx.sym.Pooling(data=body, kernel=(2, 2), stride=(2,2), pad=(0,0), pool_type='max')

#     body = conv_block(body, mFilters, (1,1), True, 'res1', False, dcn, 1) #TODO
#     body = conv_block(body, nFilters, (1,1), mFilters==nFilters, 'res2', binarize, dcn, 1) #binarize=True?

#     heatmap = None

#     for i in xrange(nStacks):
#       shortcut = body
#       if config.net_sat>0:
#         sat = SAT(body, nFilters, nModules, config.net_n+1, workspace, 'sat%d'%(i))
#         body = sat.get()
#       else:
#         body = hourglass(body, nFilters, nModules, config.net_n, workspace, 'stack%d_hg'%(i), binarize, dcn)
#       for j in xrange(nModules):
#         body = conv_block(body, nFilters, (1,1), True, 'stack%d_unit%d'%(i,j), binarize, dcn, 1)
#       _dcn = True if config.net_dcn>=2 else False
#       ll = ConvFactory(body, nFilters, (1,1), dcn = _dcn, name='stack%d_ll'%(i))
#       _name = "heatmap%d"%(i) if i<nStacks-1 else "heatmap"
#       _dcn = True if config.net_dcn>=2 else False
#       if not _dcn:
#           out = Conv(data=ll, num_filter=num_classes, kernel=(1, 1), stride=(1,1), pad=(0,0),
#                                     name=_name, workspace=workspace)
#       else:
#           out_offset = mx.symbol.Convolution(name=_name+'_offset', data = ll,
#                 num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
#           out = mx.contrib.symbol.DeformableConvolution(name=_name, data=ll, offset=out_offset,
#                 num_filter=num_classes, pad=(1,1), kernel=(3, 3), num_deformable_group=1, stride=(1, 1), dilate=(1, 1), no_bias=False)
#           #out = Conv(data=ll, num_filter=num_classes, kernel=(3,3), stride=(1,1), pad=(1,1),
#           #                          name=_name, workspace=workspace)

#       if i<nStacks-1:
#         ll2 = Conv(data=ll, num_filter=nFilters, kernel=(1, 1), stride=(1,1), pad=(0,0),
#                                   name="stack%d_ll2"%(i), workspace=workspace)
#         out2 = Conv(data=out, num_filter=nFilters, kernel=(1, 1), stride=(1,1), pad=(0,0),
#                                   name="stack%d_out2"%(i), workspace=workspace)
#         body = mx.symbol.add_n(shortcut, ll2, out2)
#         _dcn = True if (config.net_dcn==1 or config.net_dcn==3) else False
#         if _dcn:
#             _name = "stack%d_out3" % (i)
#             out3_offset = mx.symbol.Convolution(name=_name+'_offset', data = body,
#                   num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
#             out3 = mx.contrib.symbol.DeformableConvolution(name=_name, data=body, offset=out3_offset,
#                   num_filter=nFilters, pad=(1,1), kernel=(3, 3), num_deformable_group=1, stride=(1, 1), dilate=(1, 1), no_bias=False)
#             body = out3
#       elif i==nStacks-1:
#           heatmap = out
      
#       # loss = ce_loss(out, ref_label)
#       # loss = loss/nStacks
#       loss = l2_loss(out, ref_label)
#       losses.append(loss)
#       if config.net_coherent>0:
#           ux, dx = coherentor.get(out)
#           closs = l2_loss(ux, dx)
#           closs = closs/nStacks
#           closses.append(closs)

#     pred = mx.symbol.BlockGrad(heatmap)
#     #loss = mx.symbol.add_n(*losses)
#     #loss = mx.symbol.MakeLoss(loss)
#     #syms = [loss]
#     syms = []
#     for loss in losses:
#       loss = mx.symbol.MakeLoss(loss)
#       syms.append(loss)
#     if len(closses)>0:
#         coherent_weight = 0.0001
#         closs = mx.symbol.add_n(*closses)
#         closs = mx.symbol.MakeLoss(closs, grad_scale = coherent_weight)
#         syms.append(closs)
#     syms.append(pred)
#     sym = mx.symbol.Group( syms )
#     return sym


