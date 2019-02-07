import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

from subspace_dense_layer import *
from smallworld_dense_layer import *
from anneal_dense_layer import *

def network(input_data, output_dim, network_structure, anneal, is_train = False, reuse = False, use_BN = False, act = tf.nn.tanh):

    w_init = tf.random_normal_initializer(stddev=0.5)
    b_init = tf.random_normal_initializer(stddev=0.5)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("network", reuse=reuse) as vs:
        
        nn = InputLayer(input_data, name = 'input')

        information_loss = tf.zeros([1], name = 'information_loss')

        for i in range(0, len(network_structure)):
            if use_BN == True:
                # n = DenseLayer(nn, network_structure[i], act = tf.identity, W_init = w_init, b_init = b_init, name = 'layer_%d' % i)
                # n = SubSpaceWrapper(nn, n_units = network_structure[i], subDiv = 8, subConnect = 8, act = tf.identity, name = 'layer_%d' % i, W_init = w_init)
                # n = SmallWorldWrapper(nn, n_units = network_structure[i], rate = 0.99, act = tf.identity, name = 'layer_%d' % i, W_init = w_init)

                n = AnnealDenseLayer(nn, n_units = network_structure[i], sigma = anneal, act = tf.identity, name = 'layer_%d' % i, W_init = w_init, is_train = is_train)

                # Calculate Gradients for Minimize Information
                gradients = tf.gradients(n.outputs, [nn.outputs])[0]
                information_loss += tf.reduce_sum(tf.square(gradients))

                nn = n
                n = BatchNormLayer(nn, act = act, is_train = is_train, gamma_init = g_init, name = 'layer_%d/bn' % i)
                nn = n
            else:
                # n = DenseLayer(nn, network_structure[i], act = act, W_init = w_init, b_init = b_init, name = 'layer_%d' % i)
                n = AnnealDenseLayer(nn, n_units = network_structure[i], sigma = anneal, act = tf.identity, name = 'layer_%d' % i, W_init = w_init, is_train = is_train)

                # Calculate Gradients for Minimize Information
                gradients = tf.gradients(n.outputs, [nn.outputs])[0]
                information_loss += tf.reduce_sum(tf.square(gradients))

                nn = n

        # logits = DenseLayer(nn, output_dim, act = tf.identity, W_init = w_init, b_init = b_init, name = 'layer_%d' % len(network_structure))
        logits = AnnealDenseLayer(nn, output_dim, act = tf.identity, sigma = anneal, W_init = w_init, b_init = b_init, name = 'layer_%d' % len(network_structure), is_train = is_train)

        return logits, logits.outputs, information_loss

def activation_network(input, top_scope, network_structure, reuse = False, act = tf.nn.tanh, scope_prefix = ""):

    w_init = tf.random_normal_initializer(stddev=0.5)
    b_init = tf.random_normal_initializer(stddev=0.5)

    prev_scope = tf.get_variable_scope()
    # with tf.variable_scope(top_scope):
    with tf.variable_scope(scope_prefix + "activation", reuse=reuse) as vs:
        
        shape = input.shape
        print(shape)
        n = InputLayer(input)
        n = ReshapeLayer(n, [-1, 1])

        for i in range(0, len(network_structure)):
            n = DenseLayer(n, network_structure[i], act = act, W_init = w_init, b_init = b_init, name = 'layer_%d' % i)

        n = DenseLayer(n, 1, act = act, W_init = w_init, b_init = b_init, name = 'layer_%d' % len(network_structure))
        outs = ReshapeLayer(n, [-1, shape[1]])

        return outs.outputs
