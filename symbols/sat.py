from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mxnet as mx
import numpy as np
from config import config
from block import CAB, lin3


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
        bn_mom = 0.9
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