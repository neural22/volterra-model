__author__ = 'aloriga'


class InferenceOptions(object):
    path_load = ''
    # mandatory
    signal = []

    def __init__(self, **kwargs):
        for option in kwargs:
            if hasattr(self, option):
                setattr(self, option, kwargs.get(option))