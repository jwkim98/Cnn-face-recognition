import tensorflow as tf
import numpy as np

class Error(Exception):
    """wrong parameter"""
    pass

class ParameterError(Error):
    pass

#Private functions
def filter_variable(shape, name): #shape of the filter
    initial = tf.truncated_normal(shape, stddev = 0.1) #initialize
    return tf.Variable(initial, name = name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name = name)

#Public functions
def norm(input):
    return tf.contrib.layers.layer_norm(input)

def max_pool(x, size, stride, name = 'max_pool'):
    return tf.nn.max_pool(x, ksize = size, strides = stride, padding = 'SAME', name = name)

# shape is the shape of filter
def conv_layer(input, filter_shape, strides = [1,1,1,1], name = "conv"):
    filt = filter_variable(filter_shape, name = name + "_filter_variable")
    #filt = tf.Print(filt, [tf.reduce_sum(filt)], message = name)
    b = bias_variable([filter_shape[3]], name + '_bias_variable' ) #number of channels of output from conv2d
    out = tf.nn.conv2d(input, filt, strides = strides, padding = 'SAME', name = name + "_2d")
    return tf.nn.relu(out + b, name = name)

def full_layer(input, out_size, activation_func = 'relu', keep_prob = 1.0, name = 'full'):
    in_size = int(input.get_shape()[1])
    weight = filter_variable([in_size, out_size], name = name + '_weight')
    bias = bias_variable([out_size], name = name +'_bias')

    layer = tf.matmul(input,weight) + bias

    if(activation_func == 'relu'):
        activated = tf.nn.relu(layer)
    elif(activation_func == 'softmax'):
        activated = tf.nn.softmax(layer)
    elif(activation_func == 'sigmoid'):
        activated = tf.nn.sigmoid(layer)
    elif(activation_func == 'tanh'):
        activaed = tf.nn.tanh(layer)
    elif(activation_func == 'None'):
        activated = layer
    else:
        raise ParameterError

    activated_drop = tf.nn.dropout(activated, keep_prob = keep_prob, name = name)

    return activated_drop





