import tensorflow as tf

__author__ = 'aloriga'


def get_loss(volterra_model):
    """
    Return the default loss (mse)
    :param volterra_model: a model.VolterraModel instance
    :return: loss
    """
    # define placeholder for real output
    volterra_model.real_output = tf.placeholder(tf.float64, [volterra_model.batch_size], name="real_output")
    # default loss is MSE
    return tf.losses.mean_squared_error(
        volterra_model.real_output,
        volterra_model.model_output
    ) #  + volterra_model.get_l2_term()

