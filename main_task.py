import os
# access single GPU

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# -*- coding: utf-8 -*-
import argparse
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tqdm import tqdm

# from model import DeepSEA
from model_dnase import DeepSEA_DNase
from model_TFbinding import DeepSEA_tfbinding
from model_histone import DeepSEA_histone

# load data for different tasks
from loader_task import get_train_data_by_filename_for_task
from loader_task import parse_function_for_DNase, get_valid_data_dnase, get_test_data_dnase
from loader_task import parse_function_for_TFbinding, get_valid_data_tfbinding, get_test_data_tfbinding
from loader_task import parse_function_for_Histone, get_valid_data_histone, get_test_data_histone


from utils_task import plot_loss_curve
# from utils import calculate_auroc, calculate_aupr
from utils_task import create_dirs, write2txt, write2csv

np.random.seed(0)
tf.random.set_seed(0)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

BATCH_SIZE = 64

# 各个任务的数据分布
# dnase:(593355+564772)*2 = 2,316,254
# tf : (812544 + 803137)*2 = 3,231,362
# histone:(846613+812153)*2 = 3,317,532
CNT_dnase = 2316254
CNT_tf = 3231362
CNT_histone =3317532
# ==============for test=========================
# CNT_dnase = 640
# CNT_tf = 640
# CNT_histone =640

# =======dnase======================
def train_dnase():
    print("train_dnase() | gpu3| epoch 60| model-dense uint: 125 | date 05-22 : 23:55 \n")
    train_dataset = get_train_data_by_filename_for_task(BATCH_SIZE,
                                                     ['./data_task/traindata-dnase-00.tfrecord',
                                                      './data_task/traindata-dnase-01.tfrecord',
                                                      './data_task/traindata-dnase-02.tfrecord',
                                                      './data_task/traindata-dnase-03.tfrecord'],
                                                     parse_function_for_DNase)
    valid_data = get_valid_data_dnase()

    # Build the model.
    model = DeepSEA_DNase()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(momentum=0.9),
        loss=tf.keras.losses.BinaryCrossentropy())
    model.build(input_shape = (None, 1000, 4))
    model.summary()

    # Define the callbacks. (check_pointer\early_stopper\tensor_boarder)
    # For check_pointer: we save the model in SavedModel format
    # (Weights-only saving that contains model weights and optimizer status)
    check_pointer = tf.keras.callbacks.ModelCheckpoint(
        # filepath='./result/dnase/model/ckpt',
        filepath='./result/dnase/model/dnase_ckpt_{epoch:02d}.hdf5',
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        save_freq='epoch',
        load_weights_on_restart=False)
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        # patience=5,
        patience=10,
        verbose=0)
    tensor_boarder = tf.keras.callbacks.TensorBoard(
        log_dir='./result/dnase/logs')

    # Training the model.
    history = model.fit(
        train_dataset,
        epochs=60,
        steps_per_epoch=CNT_dnase/64,
        verbose=2,
        validation_data = valid_data,
        validation_steps=8000/64,
        callbacks=[check_pointer, early_stopper, tensor_boarder])

    # Plot the loss curve of training and validation, and save the loss value of training and validation.
    print('\n history dict: ', history.history)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    np.savez('./result/dnase/model_loss.npz', train_loss = train_loss, val_loss = val_loss)
    plot_loss_curve(train_loss, val_loss, './result/dnase/model_loss.jpg')

# =========tfbinding====================
def train_tf():
    print("train_tf() | gpu0| epoch 60| model-dense uint: 690 | date 05-22 : 23:56 \n")
    train_dataset = get_train_data_by_filename_for_task(BATCH_SIZE,
                                                     ['./data_task/traindata-tfbinding-00.tfrecord',
                                                      './data_task/traindata-tfbinding-01.tfrecord',
                                                      './data_task/traindata-tfbinding-02.tfrecord',
                                                      './data_task/traindata-tfbinding-03.tfrecord'],
                                                     parse_function_for_TFbinding)
    valid_data = get_valid_data_tfbinding()

    # Build the model.
    model = DeepSEA_tfbinding()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(momentum=0.9),
        loss=tf.keras.losses.BinaryCrossentropy())
    model.build(input_shape = (None, 1000, 4))
    model.summary()

    # Define the callbacks. (check_pointer\early_stopper\tensor_boarder)
    # For check_pointer: we save the model in SavedModel format
    # (Weights-only saving that contains model weights and optimizer status)
    check_pointer = tf.keras.callbacks.ModelCheckpoint(
        # filepath='./result/tf/model/ckpt',
        filepath='./result/tf/model/tfbinding_ckpt_{epoch:02d}.hdf5',
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        save_freq='epoch',
        load_weights_on_restart=False)
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=0)
    tensor_boarder = tf.keras.callbacks.TensorBoard(
        log_dir='./result/tf/logs')

    # Training the model.
    history = model.fit(
        train_dataset,
        epochs=60,
        steps_per_epoch=CNT_tf/64,
        verbose=2,
        validation_data = valid_data,
        validation_steps=8000/64,
        callbacks=[check_pointer, early_stopper, tensor_boarder])

    # Plot the loss curve of training and validation, and save the loss value of training and validation.
    print('\n history dict: ', history.history)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    np.savez('./result/tf/model_loss.npz', train_loss = train_loss, val_loss = val_loss)
    plot_loss_curve(train_loss, val_loss, './result/tf/model_loss.jpg')
    
# ========histone===================
def train_histone():
    print("train_histone() | gpu2| epoch 60| model-dense uint: 104 |date 05-22 : 23:59 \n")
    train_dataset = get_train_data_by_filename_for_task(BATCH_SIZE,
                                                     ['./data_task/traindata-histone-00.tfrecord',
                                                      './data_task/traindata-histone-01.tfrecord',
                                                      './data_task/traindata-histone-02.tfrecord',
                                                      './data_task/traindata-histone-03.tfrecord'],
                                                     parse_function_for_Histone)
    valid_data = get_valid_data_histone()

    # Build the model.
    model = DeepSEA_histone()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(momentum=0.9),
        loss=tf.keras.losses.BinaryCrossentropy())
    model.build(input_shape = (None, 1000, 4))
    model.summary()

    # Define the callbacks. (check_pointer\early_stopper\tensor_boarder)
    # For check_pointer: we save the model in SavedModel format
    # (Weights-only saving that contains model weights and optimizer status)
    check_pointer = tf.keras.callbacks.ModelCheckpoint(
        # weights.{epoch:02d-{val_loss:.2f}}.hdf5
        filepath='./result/histone/model/histone_ckpt_{epoch:02d}.hdf5',
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        save_freq='epoch',
        load_weights_on_restart=False)
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=0)
    tensor_boarder = tf.keras.callbacks.TensorBoard(
        log_dir='./result/histone/logs')

    # Training the model.
    history = model.fit(
        train_dataset,
        epochs=60,
        steps_per_epoch=CNT_histone/64,
        verbose=2,
        validation_data = valid_data,
        validation_steps=8000/64,
        callbacks=[check_pointer, early_stopper, tensor_boarder])

    # Plot the loss curve of training and validation, and save the loss value of training and validation.
    print('\n history dict: ', history.history)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    np.savez('./result/histone/model_loss.npz', train_loss = train_loss, val_loss = val_loss)
    plot_loss_curve(train_loss, val_loss, './result/histone/model_loss.jpg')


# ============================

if __name__ == '__main__':
    # Parses the command line arguments and returns as a simple namespace.
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('-e', '--exe_mode', default='train', help='The execution mode.')
    args = parser.parse_args()

    # Selecting the execution mode (keras).
    # create_dirs(['./result', './result/model'])
    if args.exe_mode == 'train_dnase':
        train_dnase()
    elif args.exe_mode == 'train_tf':
        train_tf()
    elif args.exe_mode == 'train_histone':
        train_histone()
