from model import VolterraModel

__author__ = 'aloriga'


def init_test():
    model = VolterraModel(order=3, memory=5)
    loss = model.get_loss()

init_test()