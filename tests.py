from model import VolterraModel

__author__ = 'aloriga'


def init_test():
    model = VolterraModel(order=3, memory=5)


def train_with_fake_signal():
    training_signal = [1, 2, 3, 4, 2, 7, 4]
    desired_signal = [4, 7, 9, 4, 8, 3, 1]
    model = VolterraModel(order=3, memory=2)
    model.train_with_signals(training_signal, desired_signal)


train_with_fake_signal()
