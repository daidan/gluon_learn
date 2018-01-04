# -*- coding: UTF-8 -*-

'''
Created on 2017年12月28日

@author: superhy
'''

import pip

import mxnet as mx
import mxnet.ndarray as nd

if pip.main(['show', 'mxnet-cu80']) == 0:
    print('use gpu')
else:
    print('use cpu')

_CTX = mx.gpu()

x = nd.array([1, 2, 3], ctx=_CTX)
print(x)
