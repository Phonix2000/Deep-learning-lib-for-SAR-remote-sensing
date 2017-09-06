import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from .usr_cae_utils import *
from usr_utils import *

def usr_conv_ae_sigmoid_cross_entropy(train_x, test_x, ksize=(3,3), knum =64,
                                      pool_size = (2,2), pool_strides = (2,2),
                                      noise_factor = 0.5, epochs = 10, batch_size = 128 ):
    data_shape = np.shape(train_x)
    if np.size(data_shape) == 3:
        (ntrain_samples, nrow, ncol) = data_shape[0:3]
        nchannel =1
    elif np.size(data_shape) == 4:
        (ntrain_samples, nrow, ncol, nchannel) = data_shape[0:4]
    else:
        print('Training data set size is not valid for the dunction')
        return None


    inputs_ = tf.placeholder(tf.float32, (None, nrow, nrow, nchannel), name='inputs_')
    targets_ = tf.placeholder(tf.float32, (None, nrow, nrow, nchannel), name='targets_')

    weight_1, bias_1 = usr_init_weights_bias(ksize, knum, nchannel)
    acti_1 = usr_conv_layer(inputs_, weight_1, bias_1)
    pool1 = tf.layers.max_pooling2d(acti_1, pool_size, pool_strides, padding='same')

    upsample2 = tf.image.resize_nearest_neighbor(pool1, (nrow, ncol))
    weight_logits, bias_logits = usr_init_weights_bias((3, 3), 1, 64)
    logits_ = usr_conv_layer(upsample2, weight_logits, bias_logits, act_func=None)

    outputs_ = tf.nn.sigmoid(logits_, name='outputs_')

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits_)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_cost = []
        total_batch = int(ntrain_samples / batch_size)
        print('Begin convolutional auto-encoder training, '
              'total samples %d, total batches %d' % (ntrain_samples, total_batch))

        # begin training process
        total_time = 0.0

        for epoch in range(epochs):
            start_time = time.clock()  # record the start time for each epoch
            ave_cost = 0.0
            for batch_id in range(total_batch):
                data_start = batch_id *batch_size
                data_end = (batch_id+1) * batch_size
                imgs = train_x[data_start:data_end]
                imgs = np.reshape(imgs,[batch_size, nrow,ncol, nchannel])

                # 加入噪声
                noisy_imgs = Add_noise(imgs,20)#imgs + noise_factor * np.random.randn(*imgs.shape)
                noisy_imgs = np.clip(noisy_imgs, 0., 1.)

                batch_cost, _ = sess.run([cost, optimizer],
                                         feed_dict={inputs_: noisy_imgs,
                                            targets_: imgs})
                ave_cost += batch_cost
            ave_cost /= total_batch
            end_time = time.clock()
            cost_time = end_time - start_time
            train_cost.append(ave_cost)
            print("Epoch: {}/{} ".format(epoch+1, epochs),
                  "Training loss: {:.4f},".format(ave_cost),
                  "cost time {:.4f}".format(cost_time))

        in_imgs = test_x[10:20]
        in_imgs = np.reshape(in_imgs, [10, nrow, ncol, nchannel])
        noisy_test_imgs = Add_noise(in_imgs, 20)#in_imgs + noise_factor * np.random.randn(*in_imgs.shape)
        noisy_test_imgs = np.clip(noisy_test_imgs, 0., 1.)
        reconstructed = sess.run(outputs_,feed_dict={inputs_: noisy_test_imgs.reshape((10, 28, 28, 1))})
    # show the cost curve
    plot_curve(train_cost)

    #display the reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20, 4))
    for images, row in zip([noisy_test_imgs, reconstructed], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.reshape((28, 28)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.1)
    plt.waitforbuttonpress()

    cae_weights = dict()
    cae_weights['encoder_weight'] = weight_1
    cae_weights['encoder_bias'] = bias_1
    cae_weights['decoder_weight'] = weight_logits
    cae_weights['decoder_bias'] = bias_logits

    return cae_weights,train_cost


def  usr_conv_ae_mse(train_x, test_x = None, ksize=(3,3), knum =64,
                     pool_size = (2,2), pool_strides = (2,2),
                     noise_factor = 0.5, epochs = 100,
                     batch_size = 128, test_flag = True):
    print('Training convolutional auto-encoder with mean square errors!')

    data_shape = np.shape(train_x)
    if np.size(data_shape) == 3:
        (ntrain_samples, nrow, ncol) = data_shape[0:3]
        nchannel = 1
    elif np.size(data_shape) == 4:
        (ntrain_samples, nrow, ncol, nchannel) = data_shape[0:4]
    else:
        print('Training data set size is not valid for the dunction')
        return None

    inputs_ = tf.placeholder(tf.float32, (None, nrow, nrow, nchannel), name='inputs_')
    targets_ = tf.placeholder(tf.float32, (None, nrow, nrow, nchannel), name='targets_')

    weight_1, bias_1 = usr_init_weights_bias(ksize, knum, nchannel)
    acti_1 = usr_conv_layer(inputs_, weight_1, bias_1)
    pool1 = tf.layers.max_pooling2d(acti_1, pool_size, pool_strides, padding='same')

    upsample2 = tf.image.resize_nearest_neighbor(pool1, (nrow, ncol))
    weight_logits, bias_logits = usr_init_weights_bias((3, 3), 1, 64)
    outputs_ = usr_conv_layer(upsample2, weight_logits, bias_logits,act_func= tf.nn.elu)

    loss = tf.nn.l2_loss((targets_ - outputs_))
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    train_cost = np.zeros(epochs, dtype=np.float32)
    cae_weights = dict()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_batch = int(ntrain_samples / batch_size)
        print('Begin convolutional auto-encoder training, '
              'total samples %d, total batches %d' % (ntrain_samples, total_batch))
        for epoch in range(epochs):
            start_time = time.clock()  # record the start time for each epoch
            ave_cost = 0.0
            for batch_id in range(total_batch):
                data_start = batch_id *batch_size
                data_end = (batch_id+1) * batch_size
                imgs = train_x[data_start:data_end]
                imgs = np.reshape(imgs,[batch_size, nrow,ncol, nchannel])

                # 加入噪声
                noisy_imgs = Add_noise(imgs,20)#imgs + noise_factor * np.random.randn(*imgs.shape)
                noisy_imgs = np.clip(noisy_imgs, 0., 1.)

                batch_cost, _ = sess.run([cost, optimizer],
                                         feed_dict={inputs_: noisy_imgs,
                                            targets_: imgs})
                ave_cost += batch_cost
            ave_cost /= total_batch
            end_time = time.clock()
            cost_time = end_time - start_time
            train_cost[epoch] = ave_cost
            print("Epoch: {}/{} ".format(epoch + 1, epochs),
                  "Training loss: {:.4f},".format(ave_cost),
                  "cost time {:.4f}".format(cost_time))

            #get trained weights
            en_w = sess.run(weight_1)
            en_b = sess.run(bias_1)
            de_w = sess.run(weight_logits)
            de_b = sess.run(bias_logits)

            cae_weights['encoder_weight'] = en_w
            cae_weights['encoder_bias'] = en_b
            cae_weights['decoder_weight'] = de_w
            cae_weights['decoder_bias'] = de_b

        if test_x is not None:
            print('Test convolutional auto-encoder with trained weights!')
            in_imgs = test_x[10:20]
            in_imgs = np.reshape(in_imgs, [10, nrow, ncol, nchannel])
            noisy_test_imgs = Add_noise(in_imgs, 20)  # in_imgs + noise_factor * np.random.randn(*in_imgs.shape)
            noisy_test_imgs = np.clip(noisy_test_imgs, 0., 1.)
            reconstructed = sess.run(outputs_, feed_dict={inputs_: noisy_test_imgs.reshape((10, 28, 28, 1))})
            if test_flag == True:
                # display the reconstructed images
                fig1, axes1 = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20, 4))
                for images, row in zip([noisy_test_imgs, reconstructed], axes1):
                    for img, ax in zip(images, row):
                        ax.imshow(img.reshape((28, 28)))
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        fig1.tight_layout(pad=0.1)
                        plt.waitforbuttonpress()

    return cae_weights, train_cost

