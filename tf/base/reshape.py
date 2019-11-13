import tensorflow as tf

""" 
    tf.reshape()
    -1: flatten
    或者用于推断 shape（根据tensor中的数据）
"""
b = [[2,3,4], [1,2,3]]
a = tf.constant(b, dtype=tf.float32)
with tf.Session() as sess:
    a_ = sess.run(a)
    print(a_)
    """ 推断shape """
    b_ = sess.run(tf.reshape(a, [-1, 2]))
    print(b_)
