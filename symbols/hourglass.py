from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mxnet as mx
import numpy as np
from config import config
from block import conv_block


def Conv(**kwargs):
    body = mx.sym.Convolution(**kwargs)
    return body

def Act(data, act_type, name):
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body


def hourglass(data, nFilters, nModules, n, workspace, name, binarize, dcn):
  s = 2
  _dcn = False
  up1 = data
  for i in xrange(nModules):
    up1 = conv_block(up1, nFilters, (1,1), True, "%s_up1_%d"%(name,i), binarize, _dcn, 1)
  low1 = mx.sym.Pooling(data=data, kernel=(s, s), stride=(s,s), pad=(0,0), pool_type='max')
  for i in xrange(nModules):
    low1 = conv_block(low1, nFilters, (1,1), True, "%s_low1_%d"%(name,i), binarize, _dcn, 1)
  if n>1:
    low2 = hourglass(low1, nFilters, nModules, n-1, workspace, "%s_%d"%(name, n-1), binarize, dcn)
  else:
    low2 = low1
    for i in xrange(nModules):
      low2 = conv_block(low2, nFilters, (1,1), True, "%s_low2_%d"%(name,i), binarize, _dcn, 1) #TODO
  low3 = low2
  for i in xrange(nModules):
    low3 = conv_block(low3, nFilters, (1,1), True, "%s_low3_%d"%(name,i), binarize, _dcn, 1)
  up2 = mx.symbol.UpSampling(low3, scale=s, sample_type='nearest', workspace=512, name='%s_upsampling_%s'%(name,n), num_args=1)
  return mx.symbol.add_n(up1, up2)