import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def network(input_data, output_dim, network_structure, is_train = False, reuse = False, use_BN = False, act = tf.nn.tanh):

    w_init = tf.random_normal_initializer(stddev=0.5)
    b_init = tf.random_normal_initializer(stddev=0.5)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("network", reuse=reuse) as vs:
        
        n = InputLayer(input_data, name = 'input')

        for i in range(0, len(network_structure)):
            n = DenseLayer(n, network_structure[i], act = act, W_init = w_init, b_init = b_init, name = 'layer_%d' % i)

        logits = DenseLayer(n, output_dim, act = tf.identity, W_init = w_init, b_init = b_init, name = 'layer_%d' % len(network_structure))

        return logits, logits.outputs
