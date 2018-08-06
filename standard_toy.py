import tensorflow as tf
import tensorlayer as tl
import numpy as np
import scipy
import time
import math
import argparse

import random
import sys
import os

from toy_model import network
from data_generation import *
from tensorlayer.prepro import *

from termcolor import colored, cprint

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Run simple NN test with 100 epochs and save the results.")
parser.add_argument('outpath')
parser.add_argument('-gpu', '--cuda-gpus')
parser.add_argument('-log', '--log_dir', default = 'logs\\')
parser.add_argument('-lr', '--learning-rate', type=float, default = 0.003, help='Learning rate for momentum SGD')
parser.add_argument('-alpha', '--alpha', type=float, default = 0.9, help='momentum')
parser.add_argument('-adam', '--adam', dest='adam', action='store_const', const=True, default=False, help='Use Adam instead of momentum SGD')
parser.add_argument('-bn', '--batchnorm', dest='batchnorm', action='store_const', const=True, default=False, help='Use BatchNorm or not')
parser.add_argument('-l', '--loops', type=float, default = 3.0, help='Loops count of input data')
parser.add_argument('-sig', '--noise-sigma', type=float, default = 0.01, help='Sigma of the noise distributaion')
parser.add_argument('-nn', '--nn-structure', nargs='*', type=int, help='The structure of the network, e.g. 30 20 10 8')
parser.add_argument('-u', '--unrolled', type=int, default = 5, help='Count of unrolling steps')
parser.add_argument('-avg', '--average', type=int, default = 1, help='How many times should the network try to get an average loss & acc')
parser.add_argument('-data', '--datacount', type=int, default = 65536, help='How many data should be generated in an epoch')
parser.add_argument('-bs', '--batchsize', type=int, default = 128, help='Mini-batch size')
parser.add_argument('-lrelu', dest='activation', action='store_const', const=(lambda x: tl.act.lrelu(x, 0.2)), default=tf.nn.tanh, help='Use lRelu activation instead of tanh')
parser.add_argument('-adva', '--advancedActivation', dest='advActivation', action='store_const', const=True, default=False, help='Use Advanced activation instead of lRelu or tanh')
args = parser.parse_args()

if args.adam == True:
    method_str = "Adam"
else:
    method_str = "SGD(m)-lr%.4f-m%.2f" % (args.learning_rate, args.alpha)
structure_str = '-'.join(str(e) for e in args.nn_structure)

if args.advActivation == True:
    act_str = "advanced_3"
    # args.activation = (lambda x: 0.8518565165255 * tf.exp(-2 * tf.pow(x, 2)) - 1) # normalization constant c = (sqrt(2)*pi^(3/2)) / 3, 0.8518565165255 = c * sqrt(5).
    args.activation = (lambda x: tf.abs(x) - 0.797884561) # normalization constant c = ?
elif args.activation == tf.nn.tanh:
    act_str = "tanh"
else:
    act_str = "lRelu"

name = args.outpath.split(' ')[0] + "/(avg%d)Standard-%s(%s-%s)-[lp%1.1f-sig%.2f]" % (args.average, method_str, act_str, structure_str, args.loops, args.noise_sigma)
cprint("Output Path: " + name, 'green')

os.system("del /F /Q \"" + name + "\\\"")
if not os.path.exists(name):
    os.makedirs(name)

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_gpus

input_dim = 2
output_dim = 2
network_structure = args.nn_structure

totalLoops = args.average
num_epochs = 100
batch_size = args.batchsize
learning_rate = args.learning_rate
alpha = args.alpha
dataCount = args.datacount

classification_preview_size = 128

# show the sample datas
testX, testY, testColor = gen_data(1024, args.noise_sigma, 1.0 / (math.pi * args.loops), False)
plt.figure(figsize=(8, 8))
plt.scatter(testX, testY, c = testColor)
plt.savefig(name + "/DataSample.png")
print(colored("Data Sample picture saved.", 'magenta'))
plt.clf()
# plt.show()

# array for loss and acc data
loss_epochs = np.zeros((num_epochs, 1))
acc_epochs = np.zeros((num_epochs, 1))

def train():
    input_x = tf.placeholder('float32', [batch_size, input_dim], name = "input")
    label_y = tf.placeholder('float32', [batch_size, output_dim], name = "label")

    net, logit_train = network(input_x, output_dim, network_structure, act=args.activation, is_train = True, reuse = False, use_BN = args.batchnorm)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logit_train, labels = label_y))
    n_vars = tl.layers.get_variables_with_name('network', True, True)

    if args.adam == True:
        optimizer = tf.train.AdamOptimizer()
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = alpha)
    train_op = optimizer.minimize(loss, var_list = n_vars)

    input_preview = tf.placeholder('float32', [classification_preview_size ** 2, input_dim], name = "input_preview")

    _, logit_preview = network(input_preview, output_dim, network_structure, act=args.activation, is_train = False, reuse = True, use_BN = args.batchnorm)
    preview_softmax = tf.nn.softmax(logit_preview)

    _, logit_test = network(input_x, output_dim, network_structure, act=args.activation, is_train = False, reuse = True, use_BN = args.batchnorm)
    acc, acc_op = tf.metrics.accuracy(labels = tf.argmax(label_y, 1), predictions = tf.argmax(logit_test, 1))

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    tl.layers.initialize_global_variables(sess)

    for loopCount in range(totalLoops):

        print(colored("Loop %d" % (loopCount), 'yellow'))

        sess.run(tf.local_variables_initializer())
        tl.layers.initialize_global_variables(sess)

        for idx, epoch in enumerate(gen_epochs(num_epochs, dataCount, batch_size, input_dim, output_dim, args.loops, args.noise_sigma)):
        
            training_loss = 0
            total_acc = 0
            
            for step, (_x, _y) in enumerate(epoch):
                
                # Train
                _, n_loss = sess.run([train_op, loss], feed_dict={input_x: _x, label_y: _y})
                training_loss += n_loss

                # Test
                t_x, t_y = gen_data(batch_size, args.noise_sigma, 1.0 / (math.pi * args.loops))
                _, n_acc = sess.run([acc, acc_op], feed_dict={input_x: t_x, label_y: t_y})
                total_acc += n_acc

                print(colored("Loop #%2d - " % loopCount, 'yellow') + colored("Epoch %3d, Iteration %6d: loss = %.8f, acc = %.8f" % (idx, step, n_loss, n_acc), 'cyan'))

            # Preview
            lx = np.linspace(1, -1, classification_preview_size)
            ly = np.linspace(-1, 1, classification_preview_size)
            mx, my = np.meshgrid(ly, lx)
            preview_X = np.array([mx.flatten(), my.flatten()]).T
            preview_blends = sess.run(preview_softmax, feed_dict = {input_preview: preview_X})

            color_0 = np.array([0, 0, 1])
            color_1 = np.array([1, 0, 0])

            preview_result = np.zeros((1, classification_preview_size, classification_preview_size, 3))
            for xx in range(classification_preview_size):
                for yy in range(classification_preview_size):
                    preview_result[0, xx, yy, :] = \
                        preview_blends[xx * classification_preview_size + yy, 0] * color_0 + \
                        preview_blends[xx * classification_preview_size + yy, 1] * color_1
            tl.vis.save_images(preview_result, [1, 1], name + "/Preview_Ep%d_It%d.png" % (idx, step))

            # Store data (Loss & Acc)
            training_loss /= (dataCount // batch_size)
            total_acc /= (dataCount // batch_size)

            loss_epochs[idx] += training_loss / float(totalLoops)
            acc_epochs[idx] += total_acc / float(totalLoops)

train()

# draw graph and save files
np.save(name + "/LossData.npy", loss_epochs)
np.save(name + "/AccData.npy", acc_epochs)
print(colored("Loss & Acc data saved.", 'yellow'))

plt.title("Loss")
plt.axis((0, 100, 0, 1))
plt.plot(loss_epochs)
plt.savefig(name + "/LossGraph.png")
print(colored("Loss Graph saved.", 'magenta'))
plt.clf()
# plt.show()

plt.title("Accuracy")
plt.axis((0, 100, 0, 1))
plt.plot(acc_epochs)
plt.savefig(name + "/AccuracyGraph.png")
print(colored("Accuracy Graph saved.", 'magenta'))
plt.clf()
# plt.show()
