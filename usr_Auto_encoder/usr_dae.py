import tensorflow as tf
import numpy as np
from .usr_dae_utils import *



def usr_dae_encoder(x_, en_w, en_b, act_func = tf.nn.relu):
    return usr_dae_weighted_activated(x_, en_w, en_b, act_func)


def usr_dae_decoder(en_x, de_w, de_b, act_func = tf.nn.relu):
    return usr_dae_weighted_activated(en_x, de_w, de_b, act_func)



def usr_dae_mse(train_x,test_x = None, nhidden_cells = 32, weights_regularizer = None,
                loss_function = 'mse',en_act_func = tf.nn.relu, de_act_func = tf.nn.relu,
                optimizaer = tf.train.AdadeltaOptimizer, learning_rate = 0.001, disrplay_flag = True):





