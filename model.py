import tensorflow as tf
from scipy.special import factorial
import numpy as np

__author__ = 'aloriga'


model_parameters = ['order', 'memory']


class VolterraModel(object):
    """
    This is the base class to handle Volterra models, constructor accept order and memory of the model, default values
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
        # self.inputs = {}
        self.kernels = {}
        self._init_data_structs()
        self.build_model()

    def _init_data_structs(self):
        self.inputs = tf.placeholder(tf.float32, [self.model_memory], name="input")
        for order in range(1, self.model_order + 1):
            # init kernels, first order M, second M^2, ... M^N, where M is model memory and N is model order
            dimension = [self.model_memory for _ in range(order)]
            self.kernels[order] = tf.get_variable("kernel_order_{}".format(order), shape=dimension, dtype=tf.float32, initializer=tf.random_normal_initializer())
        print("Initialized data structure for the model")

    def build_model(self):
        last_input = self.inputs
        for order in range(2, self.model_order + 1):
            order_input = tf.matmul(tf.reshape(self.inputs, [-1, 1]), tf.reshape(last_input, [1, -1]))
            order_input = tf.reshape(order_input, [self.model_memory**order])
            last_input = order_input
            # TODO: to be finished input computing for i-th order and convolutions with kernels
            print(tf.size(order_input))

    def train(self, train_x, train_y):
        pass

    def test(self, test_x, test_y):
        pass

    def predict(self, x):
        pass






