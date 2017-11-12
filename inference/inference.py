import tensorflow as tf
from .option import InferenceOptions

__author__ = 'aloriga'


def apply(volterra_model, options):
    assert isinstance(options, InferenceOptions)
    inference_graph = tf.Graph()
    with inference_graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = False
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            volterra_model.batch_size = 1
            volterra_model.build_model()
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            volterra_model.restore_model(sess, options.path_load, saver)
            output_signal = []
            print("Len output signal {}".format(len(options.signal)))
            for sample in options.signal:
                feed_dict = {
                    volterra_model.input: [sample],
                }
                output = sess.run([volterra_model.model_output], feed_dict)[0][0]
                output_signal.append(output)
            return output_signal
