import tensorflow as tf
import numpy as np
import scipy
import time
import math

import random
import sys
import os

def gen_data(size=1000):
    X = []
    Y = []
    for i in range(size):
        c = random.randint(0, 1) * 2 - 1
        x = [0, 0]
        y = [0, 0]

        # r = b*theta
        # x = b*theta*cos(theta)
        # y = b*theta*sin(theta)
        # choose b = 1 / 2pi = 0.159155
        # thus, -6pi < theta < 6pi = |x| or |y| < 3.

        theta = random.uniform(-18.84956, 18.84956) # 6 pi
        sigma = 0.5 # AGWN
        noise = np.random.normal(0, sigma, 2)

        if c > 0:
            x = [theta * math.cos(theta) + noise[0], theta * math.sin(theta) + noise[1]]
            y = [1, 0]
        else:
            x = [(-1) * theta * math.cos(theta) + noise[0], (-1) * theta * math.sin(theta) + noise[1]]
            y = [0, 1]

        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_partition_length, batch_size, numInputs], dtype=np.float32)
    data_y = np.zeros([batch_partition_length, batch_size, numOutputs], dtype=np.float32)

    for i in range(batch_partition_length):
        data_x[i] = raw_x[batch_size * i:batch_size * (i + 1)]
        data_y[i] = raw_y[batch_size * i:batch_size * (i + 1)]

    for i in range(batch_partition_length):
        x = data_x[i]
        y = data_y[i]
        yield (x, y)

def gen_epochs(n, dataCount, batch_size):
    for i in range(n):
        yield gen_batch(gen_data(dataCount), batch_size)