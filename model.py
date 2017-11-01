import tensorflow as tf
from training import training
import numpy as np

__author__ = 'aloriga'


model_parameters = ['order', 'memory']


class VolterraModel(object):
    """
    This is the base class to handle Volterra models (https://en.wikipedia.org/wiki/Volterra_series), constructor accept order and memory of the model, default values
    are 1 for both parameters.
    """

    @staticmethod
    def check_model_parameters(**kwargs):
        for param in kwargs:
            assert param in model_parameters

    def __init__(self, **kwargs):
        VolterraModel.check_model_parameters(**kwargs)
        self.model_order = kwargs.get('order', 1)
        self.model_memory = kwargs.get('memory', 1)
        self.batch_size = kwargs.get('batch_size', 16)
        self.kernels = {}
        self.outputs = {}
        self.model_output = None
        self.real_output = None

    def _init_data_structs(self):
        self.input = tf.placeholder(tf.float32, [self.batch_size, self.model_memory], name="input")
        for order in range(1, self.model_order + 1):
            # init kernels, first order M, second M^2, ... M^N, where M is model memory and N is model order
            dimension = self.model_memory ** order
            self.kernels[order] = tf.get_variable("kernel_order_{}".format(order), shape=[dimension], dtype=tf.float32, initializer=tf.random_normal_initializer())

    def build_model(self):
        self._init_data_structs()
        # compute the first order output
        self.outputs[1] = tf.reduce_sum(tf.multiply(self.input, self.kernels[1]), name="sum_{}".format(1), axis=1)
        last_input = self.input
        for order in range(2, self.model_order + 1):
            # compute the i-th order output (convolution(input, kernel i-th)
            order_input = tf.matmul(tf.reshape(self.input, [self.batch_size, -1, 1]), tf.reshape(last_input, [self.batch_size, 1, -1]))
            order_input = tf.reshape(order_input, [self.batch_size, self.model_memory**order])
            last_input = order_input
            self.outputs[order] = tf.reduce_sum(tf.multiply(order_input, self.kernels[order]), name="sum_{}".format(order), axis=1)
        self.model_output = tf.reduce_sum(list(self.outputs.values()), axis=0)
        print("Model built")

    def train(self, train_x, train_y, **kwargs):
        training_options = training.TrainingOptions(
                                                    train_x=train_x,
                                                    train_y=train_y,
                                                    **kwargs)
        training.apply(self, training_options)

    def train_with_signals(self, training_signal, desired_signal, **kwargs):
        train_x = []
        # first part of the signal, less then memory add zero padding
        for i in range(1, self.model_memory):
            partial = training_signal[:i]
            train_x.append(np.pad(partial, (self.model_memory - i, 0), 'constant', constant_values=(0, )))
        # create data set for x
        for i in range(self.model_memory, len(training_signal) + 1):
            train_x.append(training_signal[i - self.model_memory:i])
        # apply the training method
        self.train(train_x, desired_signal, **kwargs)

    def test(self, test_x, test_y):
        pass

    def predict(self, x):
        pass






