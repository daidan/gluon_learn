# -*- coding: UTF-8 -*-

'''
Created on 2017年12月28日

@author: superhy
'''

import mxnet as mx
import mxnet.ndarray as nd

_CTX = mx.gpu()

x = nd.array([1,2,3], ctx=_CTX)
print(x)