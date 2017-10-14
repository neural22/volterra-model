import tensorflow as tf
import os
import time

__author__ = 'aloriga'


class TrainingOptions(object):
    learning_rate = 1e4
    hist_grad = False
    path_save = ''
    batch_size = 16
    checkpoint_every = 10
    epochs = 100
    # mandatory
    loss = None
    train_x = []
    train_y = []

    def __init__(self, **kwargs):
        for option in kwargs:
            if hasattr(self, option):
                setattr(self, option, kwargs.get(option))


def get_training_ops(vars_to_update, options):
    assert isinstance(options, TrainingOptions)
    init_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=options.learning_rate)
    grads_and_vars = optimizer.compute_gradients(options.loss, var_list=vars_to_update, colocate_gradients_with_ops=True)
    op = optimizer.apply_gradients(grads_and_vars, global_step=init_step)
    return op, init_step, grads_and_vars


def generate_batches(train_x, train_y, batch_size):
    processed = 0
    while processed < len(train_x):
        yield(train_x[processed: processed+batch_size], train_y[processed: processed+batch_size])
        processed += batch_size


def apply(volterra_model, options):
    assert isinstance(options, TrainingOptions)
    training_graph = tf.Graph()
    with training_graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = False
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            train_op, global_step, grads_and_vars = get_training_ops(tf.global_variables())
            timestamp = str(int(time.time()))
            out_dir = os.path.join(options.path_save, "runs", timestamp)
            print("Writing to {}\n".format(out_dir))

            loss_summary = tf.summary.scalar("loss", options.loss)
            # acc_summary = tf.summary.scalar("accuracy", accuracy)
            summary_list = [loss_summary]
            # Keep track of gradient values and sparsity (optional)
            if options.hist_grad:
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)
                summary_list.append(grad_summaries_merged)

            # Train Summaries
            train_summary_op = tf.summary.merge(summary_list)
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary])  # , acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(max_to_keep=2)

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    'input': x_batch,
                    'real_output': y_batch,
                }
                _, step, summaries, train_loss = sess.run([train_op, global_step, train_summary_op, options.loss], feed_dict)
                train_summary_writer.add_summary(summaries, step)

            for epoch in range(1, options.epochs + 1):
                # TODO implement Validation Step
                for batch in generate_batches(options.train_x, options.train_y, options.batch_size):
                    if batch is None or len(batch) < options.batch_size:
                        continue
                    train_step(batch[0], batch[1])
                    current_step = tf.train.global_step(sess, global_step)
                    print("Epoch {} - Step {}".format(epoch, current_step))
                    if current_step % options.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))




