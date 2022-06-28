# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.io as sio

# np.argmax: one-hot to label
# ===========dnase=====================
def get_valid_data_dnase():
    data = sio.loadmat('./data/valid.mat')
    x = data['validxdata']  # shape = (8000, 4, 1000)
    y = data['validdata']  # shape = (8000, 919)
    x = np.transpose(x, (0, 2, 1)).astype(dtype=np.float32)  # shape = (8000, 1000, 4)
    
    y = np.transpose(y, (1, 0)).astype(dtype=np.int32)  # shape = (919, 8000)
    y_part = y[:125] # shape = (125, 8000)
    y = np.transpose(y_part, (1, 0)).astype(dtype=np.int32)  # shape = (8000, 125)
    
    return (x, y)


def get_test_data_dnase():
    filename = './data/test.mat'
    data = sio.loadmat(filename)
    x = data['testxdata']  # shape = (455024, 4, 1000)
    y = data['testdata']  # shape = (455024, 919)
    x = np.transpose(x, (0, 2, 1)).astype(np.float32)  # shape = (455024, 1000, 4)
    
    y = np.transpose(y, (1, 0)).astype(np.float32)  # shape = (919, 455024)
    y_part = y[:125]
    y = np.transpose(y_part, (1, 0)).astype(np.float32)  # shape = (455024, 125)
    return (x, y)

def parse_function_for_DNase(example_proto):
    dics = {
        'x': tf.io.FixedLenFeature([1000, 4], tf.int64),
        'y': tf.io.FixedLenFeature([919], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, dics)
    x = tf.reshape(parsed_example['x'], [1000, 4])
    y = tf.reshape(parsed_example['y'], [919])
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    
    y_part1,y_part2,y_part3 = y[:125],y[125:815],y[815:919]
#     只返回dnase标签部分的数据
    return (x, y_part1)# shape = ([1000,4],[125])
# =============TFbinding============
def get_valid_data_tfbinding():
    data = sio.loadmat('./data/valid.mat')
    x = data['validxdata']  # shape = (8000, 4, 1000)
    y = data['validdata']  # shape = (8000, 919)
    x = np.transpose(x, (0, 2, 1)).astype(dtype=np.float32)  # shape = (8000, 1000, 4)
    
    y = np.transpose(y, (1, 0)).astype(dtype=np.int32)  # shape = (919, 8000)
    y_part = y[125:815]
    y = np.transpose(y_part, (1, 0)).astype(dtype=np.int32)  # shape = (8000, 690)
    return (x, y)


def get_test_data_tfbinding():
    filename = './data/test.mat'
    data = sio.loadmat(filename)
    x = data['testxdata']  # shape = (455024, 4, 1000)
    y = data['testdata']  # shape = (455024, 919)
    x = np.transpose(x, (0, 2, 1)).astype(np.float32)  # shape = (455024, 1000, 4)
    
    y = np.transpose(y, (1, 0)).astype(np.float32)  # shape = (919,455024)
    y_part = y[125:815]
    y = np.transpose(y_part, (1, 0)).astype(np.float32)  # shape = (455024, 690)
    return (x, y)

def parse_function_for_TFbinding(example_proto):
    dics = {
        'x': tf.io.FixedLenFeature([1000, 4], tf.int64),
        'y': tf.io.FixedLenFeature([919], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, dics)
    x = tf.reshape(parsed_example['x'], [1000, 4])
    y = tf.reshape(parsed_example['y'], [919])
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    
    y_part1,y_part2,y_part3 = y[0:125],y[125:815],y[815:919]
#     只返回tf标签部分的数据
    return (x, y_part2)# shape = ([1000,4],[690])

# ==========histone================
def get_valid_data_histone():
    data = sio.loadmat('./data/valid.mat')
    x = data['validxdata']  # shape = (8000, 4, 1000)
    y = data['validdata']  # shape = (8000, 919)
    x = np.transpose(x, (0, 2, 1)).astype(dtype=np.float32)  # shape = (8000, 1000, 4)
    y = np.transpose(y, (1, 0)).astype(dtype=np.int32)  # shape = (919, 8000)
    y_part = y[815:]
    y = np.transpose(y_part, (1, 0)).astype(dtype=np.int32)  # shape = (8000, 104)
    return (x, y) 

def get_test_data_histone():
    filename = './data/test.mat'
    data = sio.loadmat(filename)
    x = data['testxdata']  # shape = (455024, 4, 1000)
    y = data['testdata']  # shape = (455024, 919)
    x = np.transpose(x, (0, 2, 1)).astype(np.float32)  # shape = (455024, 1000, 4)
    y = np.transpose(y, (1, 0)).astype(dtype=np.int32)  # shape = (919, 455024)
    y_part = y[815:]
    y = np.transpose(y_part, (1, 0)).astype(dtype=np.int32)  # shape = (455024, 104)
    return (x, y)

def parse_function_for_Histone(example_proto):
    dics = {
        'x': tf.io.FixedLenFeature([1000, 4], tf.int64),
        'y': tf.io.FixedLenFeature([919], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, dics)
    x = tf.reshape(parsed_example['x'], [1000, 4])
    y = tf.reshape(parsed_example['y'], [919])
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    
    y_part1,y_part2,y_part3 = y[0:125],y[125:815],y[815:919]
#     只返回histone标签部分的数据
    return (x, y_part3)# shape = ([1000,4],[104])

# ============================
# 各个任务的数据分布
# dnase:(593355+564772)*2 = 2,316,254
# tf : (812544 + 803137)*2 = 3,231,362
# histone:(846613+812153)*2 = 3,317,532

def get_train_data_by_filename_for_task(batch_size,path,parse_func):
    filenames = path
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=100000, num_parallel_reads=4)
    dataset = dataset.shuffle(buffer_size=10000)
    # num_parallel_calls 默认参数
    dataset = dataset.map(map_func=parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset # 4400000/64 = 68750
