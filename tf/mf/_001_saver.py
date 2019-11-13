from __future__ import print_function

import tensorflow as tf
import numpy as np


# Save to file
# remember to define the same dtype and shape when restore
def net_save():
    W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
    b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

    if int(tf.__version__.split('.')[1]) < 12 and int(tf.__version__.split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess, "my_net/save_net.ckpt")
        print("Save to path: ", save_path)


# restore variables
# redefine the same shape and same type for your variables
def net_load():
    W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
    b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

    # not need init step

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "my_net/save_net.ckpt")
        print("weights:", sess.run(W))
        print("biases:", sess.run(b))


# 这里只演示了保存和load的步骤，没有通过训练来修改weights, biases
# net_save()
net_load()
