import tensorflow as tf
import numpy as np
import scipy
import time
import math

import random
import sys
import os

def gen_data(size=1000, sigma = 0.01, b = 0.0530516, forTrain = True):
    X = []
    Y = []
    C = []
    for i in range(size):
        c = random.randint(0, 1) * 2 - 1
        x = [0, 0]
        y = [0, 0]

        # r = b*theta
        # x = b*theta*cos(theta)
        # y = b*theta*sin(theta)
        # choose b = 1 / 2pi = 0.159155
        # thus, 0 < theta < 6pi = |x| or |y| < 3.

        theta = random.uniform(0, 1.0 / b) # 6 pi
        noise = np.random.normal(0, sigma, 2) # AGWN

        if c > 0:
            x = [b * theta * math.cos(theta) + noise[0], b * theta * math.sin(theta) + noise[1]]
            y = [1, 0]
            C.append('b')
        else:
            x = [(-1) * b * theta * math.cos(theta) + noise[0], (-1) * b * theta * math.sin(theta) + noise[1]]
            y = [0, 1]
            C.append('r')

        X.append(x)
        Y.append(y)

    if forTrain:
        return np.array(X), np.array(Y)
    else:
        return np.array(X)[:, 0], np.array(X)[:, 1], C

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, input_dim, output_dim):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_partition_length, batch_size, input_dim], dtype=np.float32)
    data_y = np.zeros([batch_partition_length, batch_size, output_dim], dtype=np.float32)

    for i in range(batch_partition_length):
        data_x[i] = raw_x[batch_size * i:batch_size * (i + 1)]
        data_y[i] = raw_y[batch_size * i:batch_size * (i + 1)]

    for i in range(batch_partition_length):
        x = data_x[i]
        y = data_y[i]
        yield (x, y)

def gen_epochs(n, dataCount, batch_size, input_dim, output_dim, loops, sigma):
    for i in range(n):
        yield gen_batch(gen_data(dataCount, sigma, 1.0 / (loops * math.pi)), batch_size, input_dim, output_dim)