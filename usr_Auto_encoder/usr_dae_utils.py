import tensorflow as tf
import numpy as np


'''
    Function definitions for dae initialization
'''
def xavier_init(fan_in, fan_out, constant=1.):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

def usr_dae_init_weight(n_input, n_output):
    return tf.Variable(xavier_init(n_input, n_output))

def usr_dae_init_bias(nsize):
    return tf.Variable(tf.zeros([nsize], dtype=tf.float32))

def usr_dae_weights_init(input_size,hidden_size):
    en_w = usr_dae_init_weight(input_size,hidden_size)
    en_b = usr_dae_init_bias(hidden_size)

    de_w = usr_dae_init_weight(hidden_size,input_size)
    de_b = usr_dae_init_bias(input_size)
    return en_w,en_b,de_w,de_b

'''
    Function definition for dae processing
'''
def usr_dae_add_noise(input_data, SNR_dB =20, noise_type = 'additive'):
    SNR_amp = np.exp(SNR_dB / 10)
    data_amp = np.max(input_data)
    noise_ratio = data_amp / SNR_amp
    noise_sig = np.random.randn(*input_data.shape)
    if noise_type == 'additive':
        noised_sig = input_data + noise_ratio * noise_sig
    elif noise_type == 'multiple_add':
        noise_sig = 1 + noise_sig / noise_ratio
        noised_sig = input_data * noise_sig
    elif noise_type == 'multiple_sub':
        noise_sig = 1 - noise_sig / noise_ratio
        noised_sig = input_data * noise_sig
    else:
        print('Undefined noise type')
        noised_sig = None
    return noised_sig



def usr_dae_weighted_activated(x_, weights, biases, act_func = tf.nn.relu):
    weighted_ = tf.matmul(x_, weights)
    weighted_ = tf.nn.bias_add(weighted_, biases)
    acti_ = act_func(weighted_)
    return acti_

'''
    Function definitions for dae loss, regularization and optimization
'''

def usr_dae_mse_loss(ouputs_, targets_):
    return tf.reduce_mean(tf.nn.l2_loss(targets_-ouputs_))

def usr_dae_L1_regularizer(weights):
    return tf.reduce_mean(tf.abs(weights))

def usr_dae_L2_regularizer(weights):
    return tf.reduce_mean(tf.pow(weights,2.0))