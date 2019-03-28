from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mxnet as mx
import numpy as np
from config import config



def Conv(**kwargs):
    body = mx.sym.Convolution(**kwargs)
    return body

def Act(data, act_type, name):
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body


def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type="relu", mirror_attr={}, with_act=True, dcn=False, name=''):
    bn_mom = config.bn_mom
    workspace = config.workspace
    if not dcn:
      conv = mx.symbol.Convolution(
          data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True, workspace=workspace, name=name+'_conv')
    else:
        conv_offset = mx.symbol.Convolution(name=name+'_conv_offset', data = data,
                num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
        conv = mx.contrib.symbol.DeformableConvolution(name=name+"_conv", data=data, offset=conv_offset,
                num_filter=num_filter, pad=(1,1), kernel=(3,3), num_deformable_group=1, stride=stride, dilate=(1, 1), no_bias=False)
    bn = mx.symbol.BatchNorm(data=conv, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name+'_bn')
    if with_act:
      act = Act(bn, act_type, name=name+'_relu')
      #act = mx.symbol.Activation(
      #    data=bn, act_type=act_type, attr=mirror_attr, name=name+'_relu')
      return act
    else:
      return bn


def conv_resnet(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs):
    bit = 1
    ACT_BIT = config.ACT_BIT
    bn_mom = config.bn_mom
    workspace = config.workspace
    memonger = config.memonger
    #print('in unit2')
    # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    if not binarize:
      act1 = Act(data=bn1, act_type='relu', name=name + '_relu1')
      conv1 = Conv(data=act1, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                                 no_bias=True, workspace=workspace, name=name + '_conv1')
    else:
      act1 = mx.sym.QActivation(data=bn1, act_bit=ACT_BIT, name=name + '_relu1', backward_only=True)
      conv1 = mx.sym.QConvolution(data=act1, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                                 no_bias=True, workspace=workspace, name=name + '_conv1', act_bit=ACT_BIT, weight_bit=bit)
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    if not binarize:
      act2 = Act(data=bn2, act_type='relu', name=name + '_relu2')
      conv2 = Conv(data=act2, num_filter=int(num_filter*0.5), kernel=(3,3), stride=(1,1), pad=(1,1),
                                 no_bias=True, workspace=workspace, name=name + '_conv2')
    else:
      act2 = mx.sym.QActivation(data=bn2, act_bit=ACT_BIT, name=name + '_relu2', backward_only=True)
      conv2 = mx.sym.QConvolution(data=act2, num_filter=int(num_filter*0.5), kernel=(3,3), stride=(1,1), pad=(1,1),
                                 no_bias=True, workspace=workspace, name=name + '_conv2', act_bit=ACT_BIT, weight_bit=bit)
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    if not binarize:
      act3 = Act(data=bn3, act_type='relu', name=name + '_relu3')
      conv3 = Conv(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                 workspace=workspace, name=name + '_conv3')
    else:
      act3 = mx.sym.QActivation(data=bn3, act_bit=ACT_BIT, name=name + '_relu3', backward_only=True)
      conv3 = mx.sym.QConvolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                 no_bias=True, workspace=workspace, name=name + '_conv3', act_bit=ACT_BIT, weight_bit=bit)
    #if binarize:
    #  conv3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4')
    if dim_match:
        shortcut = data
    else:
        if not binarize:
          shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name+'_sc')
        else:
          shortcut = mx.sym.QConvolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0),
                               no_bias=True, workspace=workspace, name=name + '_sc', act_bit=ACT_BIT, weight_bit=bit)
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv3 + shortcut


def conv_hpm(data, num_filter, stride, dim_match, name, binarize, dcn, dilation, **kwargs):
    bit = 1
    ACT_BIT = config.ACT_BIT
    bn_mom = config.bn_mom
    workspace = config.workspace
    memonger = config.memonger
    #print('in unit2')
    # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    if not binarize:
      act1 = Act(data=bn1, act_type='relu', name=name + '_relu1')
      if not dcn:
          conv1 = Conv(data=act1, num_filter=int(num_filter*0.5), kernel=(3,3), stride=(1,1), pad=(dilation,dilation), dilate=(dilation,dilation),
                                     no_bias=True, workspace=workspace, name=name + '_conv1')
      else:
          conv1_offset = mx.symbol.Convolution(name=name+'_conv1_offset', data = act1,
                num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
          conv1 = mx.contrib.symbol.DeformableConvolution(name=name+'_conv1', data=act1, offset=conv1_offset,
                num_filter=int(num_filter*0.5), pad=(1,1), kernel=(3, 3), num_deformable_group=1, stride=(1, 1), dilate=(1, 1), no_bias=True)
    else:
      act1 = mx.sym.QActivation(data=bn1, act_bit=ACT_BIT, name=name + '_relu1', backward_only=True)
      conv1 = mx.sym.QConvolution_v1(data=act1, num_filter=int(num_filter*0.5), kernel=(3,3), stride=(1,1), pad=(1,1),
                                 no_bias=True, workspace=workspace, name=name + '_conv1', act_bit=ACT_BIT, weight_bit=bit)
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    if not binarize:
      act2 = Act(data=bn2, act_type='relu', name=name + '_relu2')
      if not dcn:
          conv2 = Conv(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(dilation,dilation), dilate=(dilation,dilation),
                                     no_bias=True, workspace=workspace, name=name + '_conv2')
      else:
          conv2_offset = mx.symbol.Convolution(name=name+'_conv2_offset', data = act2,
                num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
          conv2 = mx.contrib.symbol.DeformableConvolution(name=name+'_conv2', data=act2, offset=conv2_offset,
                num_filter=int(num_filter*0.25), pad=(1,1), kernel=(3, 3), num_deformable_group=1, stride=(1, 1), dilate=(1, 1), no_bias=True)
    else:
      act2 = mx.sym.QActivation(data=bn2, act_bit=ACT_BIT, name=name + '_relu2', backward_only=True)
      conv2 = mx.sym.QConvolution_v1(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                 no_bias=True, workspace=workspace, name=name + '_conv2', act_bit=ACT_BIT, weight_bit=bit)
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    if not binarize:
      act3 = Act(data=bn3, act_type='relu', name=name + '_relu3')
      if not dcn:
          conv3 = Conv(data=act3, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(dilation,dilation), dilate=(dilation,dilation), 
                  no_bias=True, workspace=workspace, name=name + '_conv3')
      else:
          conv3_offset = mx.symbol.Convolution(name=name+'_conv3_offset', data = act3,
                num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
          conv3 = mx.contrib.symbol.DeformableConvolution(name=name+'_conv3', data=act3, offset=conv3_offset,
                num_filter=int(num_filter*0.25), pad=(1,1), kernel=(3, 3), num_deformable_group=1, stride=(1, 1), dilate=(1, 1), no_bias=True)
    else:
      act3 = mx.sym.QActivation(data=bn3, act_bit=ACT_BIT, name=name + '_relu3', backward_only=True)
      conv3 = mx.sym.QConvolution_v1(data=act3, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                 no_bias=True, workspace=workspace, name=name + '_conv3', act_bit=ACT_BIT, weight_bit=bit)
    conv4 = mx.symbol.Concat(*[conv1, conv2, conv3])
    if binarize:
      conv4 = mx.sym.BatchNorm(data=conv4, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4')
    if dim_match:
        shortcut = data
    else:
        if not binarize:
          shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name+'_sc')
        else:
          #assert(False)
          shortcut = mx.sym.QConvolution_v1(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0),
                               no_bias=True, workspace=workspace, name=name + '_sc', act_bit=ACT_BIT, weight_bit=bit)
          shortcut = mx.sym.BatchNorm(data=shortcut, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv4 + shortcut
    #return bn4 + shortcut
    #return act4 + shortcut


def block17(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}, name=''):
    tower_conv = ConvFactory(net, 192, (1, 1), name=name+'_conv')
    tower_conv1_0 = ConvFactory(net, 129, (1, 1), name=name+'_conv1_0')
    tower_conv1_1 = ConvFactory(tower_conv1_0, 160, (1, 7), pad=(1, 2), name=name+'_conv1_1')
    tower_conv1_2 = ConvFactory(tower_conv1_1, 192, (7, 1), pad=(2, 1), name=name+'_conv1_2')
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
    tower_out = ConvFactory(
        tower_mixed, input_num_channels, (1, 1), with_act=False, name=name+'_conv_out')
    net = net+scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net

def block35(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}, name=''):
    M = 1.0
    tower_conv = ConvFactory(net, int(input_num_channels*0.25*M), (1, 1), name=name+'_conv')
    tower_conv1_0 = ConvFactory(net, int(input_num_channels*0.25*M), (1, 1), name=name+'_conv1_0')
    tower_conv1_1 = ConvFactory(tower_conv1_0, int(input_num_channels*0.25*M), (3, 3), pad=(1, 1), name=name+'_conv1_1')
    tower_conv2_0 = ConvFactory(net, int(input_num_channels*0.25*M), (1, 1), name=name+'_conv2_0')
    tower_conv2_1 = ConvFactory(tower_conv2_0, int(input_num_channels*0.375*M), (3, 3), pad=(1, 1), name=name+'_conv2_1')
    tower_conv2_2 = ConvFactory(tower_conv2_1, int(input_num_channels*0.5*M), (3, 3), pad=(1, 1), name=name+'_conv2_2')
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_1, tower_conv2_2])
    tower_out = ConvFactory(
        tower_mixed, input_num_channels, (1, 1), with_act=False, name=name+'_conv_out')

    net = net+scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net

def conv_inception(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs):
    assert not binarize
    if stride[0]>1 or not dim_match:
        return conv_resnet(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs)
    conv4 = block35(data, num_filter, name=name+'_block35')
    return conv4

def conv_cab(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs):
    workspace = config.workspace
    if stride[0]>1 or not dim_match:
        return conv_hpm(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs)
    cab = CAB(data, num_filter, 1, 4, workspace, name, dilate, 1)
    return cab.get()

def conv_block(data, num_filter, stride, dim_match, name, binarize, dcn, dilate):
  if config.net_block=='resnet':
    return conv_resnet(data, num_filter, stride, dim_match, name, binarize, dcn, dilate)
  elif config.net_block=='inception':
    return conv_inception(data, num_filter, stride, dim_match, name, binarize, dcn, dilate)
  elif config.net_block=='hpm':
    return conv_hpm(data, num_filter, stride, dim_match, name, binarize, dcn, dilate)
  elif config.net_block=='cab':
    return conv_cab(data, num_filter, stride, dim_match, name, binarize, dcn, dilate)


#def lin(data, num_filter, workspace, name, binarize, dcn):
#  bit = 1
#  ACT_BIT = config.ACT_BIT
#  bn_mom = config.bn_mom
#  workspace = config.workspace
#  if not binarize:
#    if not dcn:
#        conv1 = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
#                                      no_bias=True, workspace=workspace, name=name + '_conv')
#        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn')
#        act1 = Act(data=bn1, act_type='relu', name=name + '_relu')
#        return act1
#    else:
#        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn')
#        act1 = Act(data=bn1, act_type='relu', name=name + '_relu')
#        conv1_offset = mx.symbol.Convolution(name=name+'_conv_offset', data = act1,
#                num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
#        conv1 = mx.contrib.symbol.DeformableConvolution(name=name+"_conv", data=act1, offset=conv1_offset,
#                num_filter=num_filter, pad=(1,1), kernel=(3, 3), num_deformable_group=1, stride=(1, 1), dilate=(1, 1), no_bias=False)
#        #conv1 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
#        #                              no_bias=False, workspace=workspace, name=name + '_conv')
#        return conv1
#  else:
#    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn')
#    act1 = Act(data=bn1, act_type='relu', name=name + '_relu')
#    conv1 = mx.sym.QConvolution_v1(data=act1, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
#                               no_bias=True, workspace=workspace, name=name + '_conv', act_bit=ACT_BIT, weight_bit=bit)
#    conv1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
#    return conv1


def lin3(data, num_filter, workspace, name, k, g=1, d=1):
    bn_mom = config.bn_mom
    workspace = config.workspace
    if k!=3:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(k,k), stride=(1,1), pad=((k-1)//2,(k-1)//2), num_group=g,
                                      no_bias=True, workspace=workspace, name=name + '_conv')
    else:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(k,k), stride=(1,1), pad=(d,d), num_group=g, dilate=(d, d),
                                      no_bias=True, workspace=workspace, name=name + '_conv')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn')
    act1 = Act(data=bn1, act_type='relu', name=name + '_relu')
    ret = act1
    return ret


class CAB:
    def __init__(self, data, nFilters, nModules, n, workspace, name, dilate, group):
        self.data = data
        self.nFilters = nFilters
        self.nModules = nModules
        self.n = n
        self.workspace = workspace
        self.name = name
        self.dilate = dilate
        self.group = group
        self.sym_map = {}

    def get_output(self, w, h):
        key = (w, h)
        if key in self.sym_map:
            return self.sym_map[key]
        ret = None
        if h==self.n:
            if w==self.n:
                ret = (self.data, self.nFilters)
            else:
                x = self.get_output(w+1, h)
                f = int(x[1]*0.5)
                if w!=self.n-1:
                    body = lin3(x[0], f, self.workspace, "%s_w%d_h%d_1"%(self.name, w, h), 3, self.group, 1)
                else:
                    body = lin3(x[0], f, self.workspace, "%s_w%d_h%d_1"%(self.name, w, h), 3, self.group, self.dilate)
                ret = (body,f)
        else:
            x = self.get_output(w+1, h+1)
            y = self.get_output(w, h+1)
            if h%2==1 and h!=w:
                xbody = lin3(x[0], x[1], self.workspace, "%s_w%d_h%d_2"%(self.name, w, h), 3, x[1])
                #xbody = xbody+x[0]
            else:
                xbody = x[0]
            #xbody = x[0]
            #xbody = lin3(x[0], x[1], self.workspace, "%s_w%d_h%d_2"%(self.name, w, h), 3, x[1])
            if w==0:
                ybody = lin3(y[0], y[1], self.workspace, "%s_w%d_h%d_3"%(self.name, w, h), 3, self.group)
            else:
                ybody = y[0]
            ybody = mx.sym.concat(y[0], ybody, dim=1)
            body = mx.sym.add_n(xbody,ybody, name="%s_w%d_h%d_add"%(self.name, w, h))
            body = body/2
            ret = (body, x[1])
        self.sym_map[key] = ret
        return ret

    def get(self):
        return self.get_output(1, 1)[0]