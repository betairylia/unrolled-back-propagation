#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

import numpy as np

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from subspace_dense_layer import MaskedDenseLayer

__all__ = [
    'SmallWorldWrapper',
]

def SmallWorldWrapper(
    prev_layer,
    n_units=100,
    act=None,
    rate=0.1,
    W_init=tf.truncated_normal_initializer(stddev=0.1),
    b_init=tf.constant_initializer(value=0.0),
    W_init_args=None,
    b_init_args=None,
    name='dense',):
    
    n_in = prev_layer.outputs.get_shape()[-1]

    # Calculate mask matrix
    mask = np.random.binomial(1, rate, (n_in, n_units))
    return MaskedDenseLayer(prev_layer, n_units, act, mask, W_init, b_init, W_init_args, b_init_args, name)
