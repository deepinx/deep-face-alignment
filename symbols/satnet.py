from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mxnet as mx
import numpy as np
from config import config
from block import CAB, lin3, conv_block, ConvFactory
from heatmap import l2_loss, ce_loss, SymCoherent



def Conv(**kwargs):
    body = mx.sym.Convolution(**kwargs)
    return body

def Act(data, act_type, name):
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body


class SAT:
    def __init__(self, data, nFilters, nModules, n, workspace, name):
        self.data = data
        self.nFilters = nFilters
        self.nModules = nModules
        self.n = n
        self.workspace = workspace
        self.name = name
        self.sym_map = {}


    def get_conv(self, data, name, dilate=1, group=1):
        cab = CAB(data, self.nFilters, self.nModules, 4, self.workspace, name, dilate, group)
        return cab.get()

    def get_output(self, w, h):
        #print(w,h)
        assert w>=1 and w<=config.net_n+1
        assert h>=1 and h<=config.net_n+1
        s = 2
        bn_mom = config.bn_mom
        key = (w,h)
        if key in self.sym_map:
            return self.sym_map[key]
        ret = None
        if h==self.n:
            if w==self.n:
                ret = self.data,64
            else:
                x = self.get_output(w+1, h)
                body = self.get_conv(x[0], "%s_w%d_h%d_1"%(self.name, w, h))
                body = mx.sym.Pooling(data=body, kernel=(s, s), stride=(s,s), pad=(0,0), pool_type='max')
                body = self.get_conv(body, "%s_w%d_h%d_2"%(self.name, w, h))
                ret = body, x[1]//2
        else:
            x = self.get_output(w+1, h+1)
            y = self.get_output(w, h+1)

            HC = False

            if h%2==1 and h!=w:
                xbody = lin3(x[0], self.nFilters, self.workspace, "%s_w%d_h%d_x"%(self.name, w, h), 3, self.nFilters, 1)
                HC = True
                #xbody = x[0]
            else:
                xbody = x[0]
            if x[1]//y[1]==2:
                if w>1:
                    ybody = mx.symbol.Deconvolution(data=y[0], num_filter=self.nFilters, kernel=(s,s), 
                      stride=(s, s),
                      name='%s_upsampling_w%d_h%d'%(self.name,w, h),
                      attr={'lr_mult': '1.0'}, workspace=self.workspace)
                    ybody = mx.sym.BatchNorm(data=ybody, fix_gamma=False, momentum=bn_mom, eps=2e-5, name="%s_w%d_h%d_y_bn"%(self.name, w, h))
                    ybody = Act(data=ybody, act_type='relu', name="%s_w%d_h%d_y_act"%(self.name, w, h))
                else:
                    if h>=1:
                        ybody = mx.symbol.UpSampling(y[0], scale=s, sample_type='nearest', workspace=512, name='%s_upsampling_w%d_h%d'%(self.name,w, h), num_args=1)
                        ybody = self.get_conv(ybody, "%s_w%d_h%d_4"%(self.name, w, h))
                    else:
                        ybody = mx.symbol.Deconvolution(data=y[0], num_filter=self.nFilters, kernel=(s,s), 
                          stride=(s, s),
                          name='%s_upsampling_w%d_h%d'%(self.name,w, h),
                          attr={'lr_mult': '1.0'}, workspace=self.workspace)
                        ybody = mx.sym.BatchNorm(data=ybody, fix_gamma=False, momentum=bn_mom, eps=2e-5, name="%s_w%d_h%d_y_bn"%(self.name, w, h))
                        ybody = Act(data=ybody, act_type='relu', name="%s_w%d_h%d_y_act"%(self.name, w, h))
                        ybody = Conv(data=ybody, num_filter=self.nFilters, kernel=(3,3), stride=(1,1), pad=(1,1),
                                              no_bias=True, name="%s_w%d_h%d_y_conv2"%(self.name, w, h), workspace=self.workspace)
                        ybody = mx.sym.BatchNorm(data=ybody, fix_gamma=False, momentum=bn_mom, eps=2e-5, name="%s_w%d_h%d_y_bn2"%(self.name, w, h))
                        ybody = Act(data=ybody, act_type='relu', name="%s_w%d_h%d_y_act2"%(self.name, w, h))
            else:
                ybody = self.get_conv(y[0], "%s_w%d_h%d_5"%(self.name, w, h))
            #if not HC:
            if config.net_sat==2 and h==3 and w==2:
              z = self.get_output(w+1, h)
              zbody = z[0]
              zbody = mx.sym.Pooling(data=zbody, kernel=(z[1], z[1]), stride=(z[1],z[1]), pad=(0,0), pool_type='avg')
              body = xbody+ybody
              body = body/2
              body = mx.sym.broadcast_mul(body, zbody)
            else: #sat==1
              body = xbody+ybody
              body = body/2
            ret = body, x[1]

        assert ret is not None
        self.sym_map[key] = ret
        return ret

    def get(self):
        return self.get_output(1, 1)[0]


def get_symbol(num_classes):
    m = config.multiplier
    sFilters = max(int(64*m), 32)
    mFilters = max(int(128*m), 32)
    nFilters = int(256*m)

    nModules = 1
    bn_mom = config.bn_mom
    workspace = config.workspace
    nStacks = config.net_stacks
    binarize = config.net_binarize
    input_size = config.input_img_size
    label_size = config.output_label_size
    use_coherent = config.net_coherent
    use_SAT = config.net_sat
    N = config.net_n
    DCN = config.net_dcn
    per_batch_size = config.per_batch_size
    print('binarize', binarize)
    print('use_coherent', use_coherent)
    print('use_SAT', use_SAT)
    print('use_N', N)
    print('use_DCN', DCN)
    print('per_batch_size', per_batch_size)
    #assert(label_size==64 or label_size==32)
    #assert(input_size==128 or input_size==256)
    coherentor = SymCoherent(per_batch_size)
    D = input_size // label_size
    print(input_size, label_size, D)
    data = mx.sym.Variable(name='data')
    data = data-127.5
    data = data*0.0078125
    gt_label = mx.symbol.Variable(name='softmax_label')
    losses = []
    closses = []
    ref_label = gt_label
    if D==4:
      body = Conv(data=data, num_filter=sFilters, kernel=(7, 7), stride=(2,2), pad=(3, 3),
                              no_bias=True, name="conv0", workspace=workspace)
    else:
      body = Conv(data=data, num_filter=sFilters, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                              no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = Act(data=body, act_type='relu', name='relu0')

    dcn = False
    body = conv_block(body, mFilters, (1,1), sFilters==mFilters, 'res0', False, dcn, 1)

    body = mx.sym.Pooling(data=body, kernel=(2, 2), stride=(2,2), pad=(0,0), pool_type='max')

    body = conv_block(body, mFilters, (1,1), True, 'res1', False, dcn, 1) #TODO
    body = conv_block(body, nFilters, (1,1), mFilters==nFilters, 'res2', binarize, dcn, 1) #binarize=True?

    heatmap = None

    for i in xrange(nStacks):
      shortcut = body
      if config.net_sat>0:
        sat = SAT(body, nFilters, nModules, config.net_n+1, workspace, 'sat%d'%(i))
        body = sat.get()
      else:
        body = hourglass(body, nFilters, nModules, config.net_n, workspace, 'stack%d_hg'%(i), binarize, dcn)
      for j in xrange(nModules):
        body = conv_block(body, nFilters, (1,1), True, 'stack%d_unit%d'%(i,j), binarize, dcn, 1)
      _dcn = True if config.net_dcn>=2 else False
      ll = ConvFactory(body, nFilters, (1,1), dcn = _dcn, name='stack%d_ll'%(i))
      _name = "heatmap%d"%(i) if i<nStacks-1 else "heatmap"
      _dcn = True if config.net_dcn>=2 else False
      if not _dcn:
          out = Conv(data=ll, num_filter=num_classes, kernel=(1, 1), stride=(1,1), pad=(0,0),
                                    name=_name, workspace=workspace)
      else:
          out_offset = mx.symbol.Convolution(name=_name+'_offset', data = ll,
                num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
          out = mx.contrib.symbol.DeformableConvolution(name=_name, data=ll, offset=out_offset,
                num_filter=num_classes, pad=(1,1), kernel=(3, 3), num_deformable_group=1, stride=(1, 1), dilate=(1, 1), no_bias=False)
          #out = Conv(data=ll, num_filter=num_classes, kernel=(3,3), stride=(1,1), pad=(1,1),
          #                          name=_name, workspace=workspace)

      if i<nStacks-1:
        ll2 = Conv(data=ll, num_filter=nFilters, kernel=(1, 1), stride=(1,1), pad=(0,0),
                                  name="stack%d_ll2"%(i), workspace=workspace)
        out2 = Conv(data=out, num_filter=nFilters, kernel=(1, 1), stride=(1,1), pad=(0,0),
                                  name="stack%d_out2"%(i), workspace=workspace)
        body = mx.symbol.add_n(shortcut, ll2, out2)
        _dcn = True if (config.net_dcn==1 or config.net_dcn==3) else False
        if _dcn:
            _name = "stack%d_out3" % (i)
            out3_offset = mx.symbol.Convolution(name=_name+'_offset', data = body,
                  num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
            out3 = mx.contrib.symbol.DeformableConvolution(name=_name, data=body, offset=out3_offset,
                  num_filter=nFilters, pad=(1,1), kernel=(3, 3), num_deformable_group=1, stride=(1, 1), dilate=(1, 1), no_bias=False)
            body = out3
      elif i==nStacks-1:
          heatmap = out
      
      # loss = ce_loss(out, ref_label)
      # loss = loss/nStacks
      loss = l2_loss(out, ref_label)
      losses.append(loss)
      if config.net_coherent>0:
          ux, dx = coherentor.get(out)
          closs = l2_loss(ux, dx)
          closs = closs/nStacks
          closses.append(closs)

    pred = mx.symbol.BlockGrad(heatmap)
    #loss = mx.symbol.add_n(*losses)
    #loss = mx.symbol.MakeLoss(loss)
    #syms = [loss]
    syms = []
    for loss in losses:
      loss = mx.symbol.MakeLoss(loss)
      syms.append(loss)
    if len(closses)>0:
        coherent_weight = 0.0001
        closs = mx.symbol.add_n(*closses)
        closs = mx.symbol.MakeLoss(closs, grad_scale = coherent_weight)
        syms.append(closs)
    syms.append(pred)
    sym = mx.symbol.Group( syms )
    return sym