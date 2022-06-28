import os
# access single GPU

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import nest_asyncio

nest_asyncio.apply()

import collections
import tensorflow_federated as tff

import argparse
import numpy as np
import tensorflow as tf

import functools
import attr

from typing import Callable
from tensorflow import keras
from tqdm import tqdm

from model import DeepSEA
from loader import get_train_data, get_valid_data, get_test_data
from utils import plot_loss_curve, plot_roc_curve, plot_pr_curve
from utils import calculate_auroc, calculate_aupr
from utils import create_dirs, write2txt, write2csv

np.random.seed(0) # ？随机种子决定处理顺序?
tf.random.set_seed(0)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

print(tff.federated_computation(lambda: "Hello, World! 5-10 | fedavg-new-api | gpu0")())  # federated_computation装饰器


def parse_function(example_proto):
    dics = {
        'x': tf.io.FixedLenFeature([1000, 4], tf.int64),
        'y': tf.io.FixedLenFeature([919], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, dics)
    x = tf.reshape(parsed_example['x'], [1000, 4])
    y = tf.reshape(parsed_example['y'], [919])
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    return (x, y)

def get_train_data_by_filename(batch_size,path):
    filenames = path
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=100000, num_parallel_reads=4)
    dataset = dataset.shuffle(buffer_size=10000)
    # num_parallel_calls 默认参数
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset # 4400000/64 = 68750


#  cpu -> gpu 加载数据

BATCH_SIZE = 64
# task1: dnase
train_dataset1 = get_train_data_by_filename(BATCH_SIZE,['./data_task/traindata-dnase-00.tfrecord','./data_task/traindata-dnase-01.tfrecord'])
train_dataset2 = get_train_data_by_filename(BATCH_SIZE,['./data_task/traindata-dnase-02.tfrecord','./data_task/traindata-dnase-03.tfrecord'])
# task2: tfbinding
train_dataset3 = get_train_data_by_filename(BATCH_SIZE,['./data_task/traindata-tfbinding-00.tfrecord','./data_task/traindata-tfbinding-01.tfrecord'])
train_dataset4 = get_train_data_by_filename(BATCH_SIZE,['./data_task/traindata-tfbinding-02.tfrecord','./data_task/traindata-tfbinding-03.tfrecord'])
# task3:histone
train_dataset5 = get_train_data_by_filename(BATCH_SIZE,['./data_task/traindata-histone-00.tfrecord','./data_task/traindata-histone-01.tfrecord'])
train_dataset6 = get_train_data_by_filename(BATCH_SIZE,['./data_task/traindata-histone-02.tfrecord','./data_task/traindata-histone-03.tfrecord'])


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
        input_spec=train_dataset1.element_spec, # 注意这里是train_dataset1！！！
        loss=tf.keras.losses.BinaryCrossentropy(), 
        metrics=[tf.keras.metrics.CategoricalAccuracy()],) 

federated_train_data = [
    train_dataset1,
    train_dataset2,
    train_dataset3,
    train_dataset4,
    train_dataset5,
    train_dataset6,
]
#  ============================================
from typing import Callable, Optional, Union

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.algorithms import aggregation
from tensorflow_federated.python.learning.metrics import aggregator as metric_aggregator
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import finalizers
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.learning.templates import model_delta_client_work

DEFAULT_SERVER_OPTIMIZER_FN = lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
#   注意：默认的服务器优化器函数是 `tf.keras.optimizers.SGD`
#   学习率为 1.0，对应于将模型增量添加到当前的服务器模型。

def build_weighted_fed_avg(
    model_fn: Callable[[], model_lib.Model],
    client_optimizer_fn: Union[optimizer_base.Optimizer,
                               Callable[[], tf.keras.optimizers.Optimizer]],
    server_optimizer_fn: Union[optimizer_base.Optimizer, Callable[
        [], tf.keras.optimizers.Optimizer]] = DEFAULT_SERVER_OPTIMIZER_FN,
    client_weighting: Optional[
        client_weight_lib.ClientWeighting] = client_weight_lib.ClientWeighting
    .NUM_EXAMPLES,
    model_distributor: Optional[distributors.DistributionProcess] = None,
    model_aggregator: Optional[factory.WeightedAggregationFactory] = None,
    metrics_aggregator: Optional[Callable[[
        model_lib.MetricFinalizersType, computation_types.StructWithPythonType
    ], computation_base.Computation]] = None,
    use_experimental_simulation_loop: bool = False
) -> learning_process.LearningProcess:
  py_typecheck.check_callable(model_fn)
  py_typecheck.check_type(client_weighting, client_weight_lib.ClientWeighting)

  @computations.tf_computation()
  def initial_model_weights_fn():
    return model_utils.ModelWeights.from_model(model_fn())

  model_weights_type = initial_model_weights_fn.type_signature.result

  if model_distributor is None:
    model_distributor = distributors.build_broadcast_process(model_weights_type)

  if model_aggregator is None:
    model_aggregator = mean.MeanFactory()
  py_typecheck.check_type(model_aggregator, factory.WeightedAggregationFactory)
  aggregator = model_aggregator.create(model_weights_type.trainable,
                                       computation_types.TensorType(tf.float32))
  process_signature = aggregator.next.type_signature
  input_client_value_type = process_signature.parameter[1]
  result_server_value_type = process_signature.result[1]
  if input_client_value_type.member != result_server_value_type.member:
    raise TypeError('`model_aggregator` does not produce a compatible '
                    '`AggregationProcess`. The processes must retain the type '
                    'structure of the inputs on the server, but got '
                    f'{input_client_value_type.member} != '
                    f'{result_server_value_type.member}.')

  if metrics_aggregator is None:
    metrics_aggregator = metric_aggregator.sum_then_finalize
  client_work = model_delta_client_work.build_model_delta_client_work(
      model_fn=model_fn,
      optimizer=client_optimizer_fn,
      client_weighting=client_weighting,
      metrics_aggregator=metrics_aggregator,
      use_experimental_simulation_loop=use_experimental_simulation_loop)
  finalizer = finalizers.build_apply_optimizer_finalizer(
      server_optimizer_fn, model_weights_type)
  return composers.compose_learning_process(initial_model_weights_fn,
                                            model_distributor, client_work,
                                            aggregator, finalizer)

iterative_process = build_weighted_fed_avg(
    model_fn,  # 参数需要是一个构造函数（如model_fn 上面的），而不是一个已经构造的实例
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(momentum=0.9),
    
)


logdir = "/tmp/logs/scalars/training/0510"
summary_writer = tf.summary.create_file_writer(logdir)
state = iterative_process.initialize()

NUM_ROUNDS = 600
with summary_writer.as_default():
  for round_num in range(1, NUM_ROUNDS):
    learning_process_output = iterative_process.next(state, federated_train_data)
    state, metrics = learning_process_output.state, learning_process_output.metrics
    train_metrics = metrics['client_work']['train']
    print("round {:2d}, train_metrics={}".format(round_num,train_metrics))
    for name, value in train_metrics.items():
      tf.summary.scalar(name, value, step=round_num)
    
    gobal_model = state.global_model_weights
    keras_model1 = create_keras_model()
    keras_model1.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])
    tf.nest.map_structure(
        lambda var, t: var.assign(t),
        keras_model1.trainable_weights, gobal_model.trainable)
    
    save_path = "./tff_result_510_ckpt/tff_alldata_"+str(round_num)+".h5"
    # save_path = "./tff_result_506_ckpt/take64_"+str(round_num)+".h5"
    keras_model1.save(save_path)