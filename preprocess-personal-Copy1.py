import os
# access single GPU

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import h5py
import numpy as np
import tensorflow as tf
import scipy.io as sio

from tqdm import tqdm

def serialize_example(x, y):
    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    example = {
        'x': tf.train.Feature(int64_list=tf.train.Int64List(value=x.flatten())),
        'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y.flatten()))}

    # Create a Features message using tf.train.Example.
    example = tf.train.Features(feature=example)
    example = tf.train.Example(features=example)
    serialized_example = example.SerializeToString()
    return serialized_example


print("read start\n")
filename = './data/train.mat'
with h5py.File(filename, 'r') as file:
    x = file['trainxdata'] # shape = (1000, 4, 4400000)
    y = file['traindata'] # shape = (919, 4400000)
    x = np.transpose(x, (2, 0, 1)) # shape = (4400000, 1000, 4)
    y = np.transpose(y, (1, 0)) # shape = (4400000, 919)
print("read finish\n")

print("task1\n")
for file_num in range(4):
    with tf.io.TFRecordWriter('./data_task_personal/traindata-tfbinding-%.2d.tfrecord' % file_num) as writer:
        for i in tqdm(range(file_num*1100000, (file_num+1)*1100000), desc="Processing Train Data {}".format(file_num), ascii=True):
            example_proto = serialize_example(x[i], y[i][125:815])
            for index in range(125,815):
                if y[i][index] == 1:
                    writer.write(example_proto)
                    break

print("task2\n")
for file_num in range(4):
    with tf.io.TFRecordWriter('./data_task_personal/traindata-histone-%.2d.tfrecord' % file_num) as writer:
        for i in tqdm(range(file_num*1100000, (file_num+1)*1100000), desc="Processing Train Data {}".format(file_num), ascii=True):
            example_proto = serialize_example(x[i], y[i][815:])
            for index in range(815,919):
                if y[i][index] == 1:
                    writer.write(example_proto)
                    break