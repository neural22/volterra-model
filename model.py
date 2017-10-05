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
        self.inputs = {}
        self.kernels = {}
        self._init_data_structs()
        self.feed_inputs = {}

    def _init_data_structs(self):
        for order in range(self.model_order):
            # compute the dimension of the i-th order of the kernel and input
            # like: first order vector has M coefficients, the second order vector M(M + 1)/2 coefficients and the
            # third order vector M (M + 1)(M + 2)/6 coefficients
            dimension = np.prod([(self.model_memory + o) for o in range(order + 1)]) / factorial(order + 1).item()
            # dimension = np.ceil(dimension)
            self.inputs[order] = tf.placeholder(tf.float32, [dimension], name="input_order_{}".format(order + 1))
            self.kernels[order] = tf.placeholder(tf.float32, [dimension], name="kernel_order_{}".format(order + 1))
            print("Kernel dimension for order {} is {}".format(order, dimension))
        print("Initialized data structure for the model")

    def set_training_input(self, data_x):
        """
        This method must be called before train the model, prepare the input for the various order
        :param data_x: list , it represents the input signal
        :return:
        """
        last_processed_index = 0
        self.feed_inputs[1] = []
        # input for first order
        while self.model_memory + last_processed_index < len(data_x):
            slice_end = self.model_memory + last_processed_index
            self.feed_inputs[1].append(np.float32(data_x[last_processed_index:slice_end]))
            last_processed_index += 1

        for order in range(2, self.model_order + 1):
            self.feed_inputs[order] = []

        for order in range(2, self.model_order + 1):
            for x_init, x in zip(self.feed_inputs[1], self.feed_inputs[order - 1]):
                matrix = np.triu(x_init.reshape(-1, 1) * x) # extract the triangular upper part of matrix returned by x^T * x
                x_order = matrix.ravel()[np.flatnonzero(matrix)]
                self.feed_inputs[order].append(x_order)

        # TODO there is a bug after the second order, dimensions don't match
        print(np.size(self.feed_inputs[1][0]))
        print(np.size(self.feed_inputs[2][0]))
        print(np.size(self.feed_inputs[3][0]))

        def train(train_x, train_y):
            pass

        def test(test_x, test_y):
            pass

        def predict(x):
            pass






