# -*- coding: UTF-8 -*-

'''
Created on 2017年12月28日

@author: superhy
'''

import pip
import mxnet as mx

def get_ctx():
    if pip.main(['show', 'mxnet-cu80']) == 0:
        _CTX = mx.gpu()
    else:
        _CTX = mx.cpu()
    
    return _CTX
