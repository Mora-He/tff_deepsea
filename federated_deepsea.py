import os
# access single GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import nest_asyncio

nest_asyncio.apply()
import collections

import tensorflow_federated as tff
import argparse
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tqdm import tqdm

from model import DeepSEA
# from loader import get_train_data, get_valid_data, get_test_data
from utils import plot_loss_curve, plot_roc_curve, plot_pr_curve
from utils import calculate_auroc, calculate_aupr
from utils import create_dirs, write2txt, write2csv

np.random.seed(0) # ？随机种子决定处理顺序?
tf.random.set_seed(0)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def parse_function(example_proto):
    dics = {
        'x': tf.io.FixedLenFeature([1000, 4], tf.int64),
        'y': tf.io.FixedLenFeature([919], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, dics)
    x = tf.reshape(parsed_example['x'], [1000, 4])
    y = tf.reshape(parsed_example['y'], [919])
#     y = tf.reshape(tf.where(tf.equal(parsed_example['y'],1)), [1,-1])
#     y = tf.squeeze(tf.reshape(tf.where(tf.equal(parsed_example['y'],1)), [1,-1]), 1)
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    return [x, y]

def get_train_data(batch_size):
    filenames = ['./data/traindata-00.tfrecord'  
#                  , './data/traindata-01.tfrecord'
#                  './data/traindata-02.tfrecord', './data/traindata-03.tfrecord'
                ]
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=100000, num_parallel_reads=4)
    dataset = dataset.shuffle(buffer_size=10000)
    # num_parallel_calls 默认参数
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset # 4400000/64 = 68750

def get_train_data_by_filename(batch_size,path):
#     filenames = [path]
    filenames = path
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=100000, num_parallel_reads=4)
    dataset = dataset.shuffle(buffer_size=10000)
    # num_parallel_calls 默认参数
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset # 4400000/64 = 68750

BATCH_SIZE = 64
# valid_data = get_valid_data()
train_dataset = get_train_data(BATCH_SIZE)

train_dataset1 = get_train_data_by_filename(BATCH_SIZE,['./data_6/traindata-00.tfrecord','./data_6/traindata-03.tfrecord'])
train_dataset2 = get_train_data_by_filename(BATCH_SIZE,['./data_6/traindata-01.tfrecord','./data_6/traindata-04.tfrecord'])
train_dataset3 = get_train_data_by_filename(BATCH_SIZE,['./data_6/traindata-02.tfrecord','./data_6/traindata-05.tfrecord'])
# train_dataset4 = get_train_data_by_filename(BATCH_SIZE,'./data_6/traindata-03.tfrecord')
# train_dataset5 = get_train_data_by_filename(BATCH_SIZE,'./data_6/traindata-04.tfrecord')
# train_dataset6 = get_train_data_by_filename(BATCH_SIZE,'./data_6/traindata-05.tfrecord')


def create_keras_model():      
  return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape = (1000, 4, )),
      tf.keras.layers.Conv1D(
            filters=320,
            kernel_size=8,
            strides=1,
            use_bias=False,
            padding='SAME',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(5e-07),
            kernel_constraint=tf.keras.constraints.MaxNorm(0.9)),
      tf.keras.layers.MaxPool1D(
            pool_size=4,
            strides=4,
            padding='SAME'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Conv1D(
            filters=480,
            kernel_size=8,
            strides=1,
            use_bias=False,
            padding='SAME',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(5e-07),
            kernel_constraint=tf.keras.constraints.MaxNorm(0.9)),
      tf.keras.layers.MaxPool1D(
            pool_size=4,
            strides=4,
            padding='SAME'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Conv1D(
            filters=960,
            kernel_size=8,
            strides=1,
            use_bias=False,
            padding='SAME',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(5e-07),
            kernel_constraint=tf.keras.constraints.MaxNorm(0.9)),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
            units=925,
            use_bias=False,
            activation='relu',
#             activity_regularizer=tf.keras.regularizers.l1(1e-08),
            kernel_regularizer=tf.keras.regularizers.l2(5e-07),
            kernel_constraint=tf.keras.constraints.MaxNorm(0.9)),
      tf.keras.layers.Dense(
            units=919,
            use_bias=False,
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l2(5e-07),
            kernel_constraint=tf.keras.constraints.MaxNorm(0.9)),
  ])

def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=train_dataset.element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(), # 
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], # 导致类型错误
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    ) 

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,  # 参数需要是一个构造函数（如model_fn 上面的），而不是一个已经构造的实例
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02)
    
    #use_experimental_simulation_loop=True
)


state = iterative_process.initialize()
federated_train_data = [train_dataset1, train_dataset2,train_dataset3
#                         , train_dataset4,train_dataset5, train_dataset6
                       ]

NUM_ROUNDS = 60
for round_num in range(1, NUM_ROUNDS):
    state, metrics = iterative_process.next(state, federated_train_data)
    print("round {:2d}, metrics={}".format(round_num, metrics))
    keras_model=create_keras_model()
    state.model.assign_weights_to(keras_model)
    keras_model.save("./tff_result_0406_batchsize32/test_"+str(round_num)+".h5")

