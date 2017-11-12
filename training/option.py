from .loss import get_loss

__author__ = 'aloriga'


class TrainingOptions(object):
    learning_rate = 1e-2
    hist_grad = False
    path_save = ''
    batch_size = 2
    checkpoint_every = 500
    epochs = 100
    # mandatory
    train_x = []
    train_y = []
    # not mandatory for training
    validation_x = []
    validation_y = []
    # validation steps must be done every n steps
    validation_every = 50

    _computed_loss = None
    print_loss_every = 50
    max_to_keep = 2

    def __init__(self, **kwargs):
        for option in kwargs:
            if hasattr(self, option):
                setattr(self, option, kwargs.get(option))

    # override this function with another loss function, should receive the model as param
    def loss(self, model):
        if self._computed_loss is None:
            self._computed_loss = get_loss(model)
        return self._computed_loss
