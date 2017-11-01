import tensorflow as tf


def get_loss(volterra_model):
    """
    Return the default loss (softmax_cross_entropy)
    :param volterra_model: a model.VolterraModel instance
    :return: loss
    """
    # define placeholder for real output
    volterra_model.real_output = tf.placeholder(tf.float32, [volterra_model.batch_size], name="real_output")
    # default loss is MSE
    return tf.nn.softmax_cross_entropy_with_logits(labels=volterra_model.real_output,
                                                   logits=volterra_model.model_output)
    # return tf.metrics.mean_squared_error(volterra_model.real_output, volterra_model.model_output)[0]
