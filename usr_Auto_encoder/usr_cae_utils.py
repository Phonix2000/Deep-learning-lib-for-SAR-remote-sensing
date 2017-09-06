import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''
    Function defined for denoising convolutional auto-encoder
'''
def usr_conv_layer(input_, weight, bias, strides = (1,1,1,1), padding = 'SAME',
                   act_func = tf.nn.relu, NAME = None):
    conv_rs = tf.nn.conv2d(input=input_, filter=weight, strides=list(strides), padding=padding)
    conv_rs = tf.nn.bias_add(conv_rs, bias, name=NAME)
    if act_func is not None:
        acti_rs = act_func(conv_rs)
    else:
        acti_rs = conv_rs
    return acti_rs

def usr_max_pooling(acti_x, ksize =(2,2), strides = (2,2), padding ='same'):
    return tf.layers.max_pooling2d(acti_x, ksize, strides, padding=padding)

def usr_init_weights_bias(ksize, knum,nchannel,stddev =0.1):
    kshape = [ksize[0], ksize[1],nchannel, knum]
    weights = tf.Variable(tf.truncated_normal(shape=kshape, stddev=0.1))
    biases = tf.Variable(tf.constant(0.0, shape=[knum], name='bias_1'))
    return weights, biases



def Add_noise(input_data, SNR_dB =20, noise_type = 'additive'):
    SNR_amp = np.exp(SNR_dB/10)
    data_amp = np.max(input_data)
    noise_ratio = data_amp/SNR_amp
    noise_sig = np.random.randn(*input_data.shape)
    if noise_type == 'additive':
        noised_sig = input_data + noise_ratio * noise_sig
    elif noise_type == 'multiple_add':
        noise_sig = 1 +  noise_sig/noise_ratio
        noised_sig = input_data * noise_sig
    elif noise_type == 'multiple_sub':
        noise_sig = 1 - noise_sig/noise_ratio
        noised_sig = input_data * noise_sig
    else:
        print('Undefined noise type')
        noised_sig =  None
    return  noised_sig


