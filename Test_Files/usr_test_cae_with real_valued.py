import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from usr_Auto_encoder import *

print("TensorFlow Version: %s" % tf.__version__)

'''
Test program for convolutional auto-encoder, if you want to run this program, plsease copy it to the upper folder
'''

mnist = input_data.read_data_sets('MNIST_data', validation_size=0, one_hot=False)

img = mnist.train.images[20]
plt.imshow(img.reshape((28, 28)))

mnist = input_data.read_data_sets('MNIST_data',one_hot= True)
train_x,test_x = mnist.train.images,mnist.test.images
#train_x = train_x.astype('float32')/255.
#test_x = test_x.astype('float32')/255.


train_x = np.reshape(train_x,[train_x.shape[0],28,28])
test_x = np.reshape(test_x, [test_x.shape[0],28,28])

#en_w,en_b,de_w,de_b= usr_conv_ae_sigmoid_cross_entropy(train_x,test_x)


my_cae_weights, train_cost = usr_conv_ae_mse(train_x, test_x, epochs= 10, test_flag = False)

print(train_cost)
print(type(train_cost))

print(my_cae_weights)


plot_curve(train_cost,fig_id= 2)
plt.waitforbuttonpress()


