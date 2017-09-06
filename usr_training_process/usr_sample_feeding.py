import tensorflow as tf
import numpy as np

def usr_make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).

    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.

    # Returns
        A list of tuples of array indices.
    """
    num_batches = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)]

def Init_shffle_indx(num_train_samples,batch_size):
    index_array = np.arange(num_train_samples)
    batches = usr_make_batches(num_train_samples, batch_size)
    return index_array, batches

def train_sample_shuffle(index_array):
    return np.random.shuffle(index_array)

def get_next_shuffle_batch(train_x, batches, batch_id):
    data_index = batches[batch_id]
    return train_x[data_index]


