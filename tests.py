from model import VolterraModel

__author__ = 'aloriga'


def init_test():
    model = VolterraModel(order=3, memory=3)
    model.set_training_input([1, 1, 2, 3, 5, 8, 13, 21])

init_test()