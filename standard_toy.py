import tensorflow as tf
import numpy as np
import scipy
import time
import math

import random
import sys
import os

from toy_model import network
from data_generation import *
from tensorlayer.prepro import *

from termcolor import colored, cprint

name = sys.argv[1]

input_dim = 2
output_dim = 2
network_structure = [8, 8, 8, 8, 8]

batch_size = 32
learning_rate = 0.01
dataCount = 100000

classification_preview_size = 128

def train():
    input_x = tf.placeholder('float32', [batch_size, input_dim], name = "input")
    label_y = tf.placeholder('float32', [batch_size, output_dim], name = "label")

    net, logits = network(input_x, output_dim, network_structure, is_train = True, reuse = False)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = label_y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    input_preview = tf.placeholder('float32', [classification_preview_size ** 2, input_dim], name = "input_preview")

    _, logit_preview = network(input_preview, output_dim, network_structure, is_train = True, reuse = True)
    preview_softmax = tf.nn.softmax(logit_preview)

    _, logit_test = network(input_x, output_dim, network_structure, is_train = True, reuse = True)
    acc, acc_op = tf.metrics.accuracy(labels = tf.argmax(label_y, 1), predictions = tf.argmax(logit_test, 1))

    sess = tf.Session()
    tl.layers.initialize_global_variables(sess)

    for idx, epoch in enumerate(gen_epochs(num_epochs, dataCount, batch_size)):
        
        training_loss = 0
        
        for step, (_x, _y) in enumerate(epoch):
            
            # Train
            _, loss = sess.run([train_op, loss], feed_dict={input_x: _x, label_y: _y})
            training_loss += loss

            # Test
            t_x, t_y = gen_data(batch_size)
            _, acc = sess.run([acc, acc_op], feed_dict={input_x: t_x, label_y: t_y})

            print(colored("Epoch %3d, Iteration %6d: loss = %.8f, acc = %.8f" % (idx, step, loss, acc), 'cyan'))

            # Preview
            lx = np.linspace(-3, 3, classification_preview_size)
            ly = np.linspace(-3, 3, classification_preview_size)
            mx, my = np.meshgrid(lx, ly)
            preview_X = np.array([mx.flatten(), my.flatten()]).T
            preview_blends = sess.run(preview_softmax, feed_dict = {input_preview: preview_X})

            preview_result = preview_blends[:, 0] * [1, 0, 0] + preview_blends[:, 1] * [0, 0, 1]
            tl.vis.save_images(preview_result, [1, 1], name + '/Preview_%d.png' % (idx * (dataCount // batch_size)) + step)

train()
