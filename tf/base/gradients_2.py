import tensorflow as tf


""" grad_ys相当于给ys做了一个乘积，也相当于将结果放大那么多倍 """


w1 = tf.get_variable('w1', shape=[3])
w2 = tf.get_variable('w2', shape=[3])

w3 = tf.get_variable('w3', shape=[3])
w4 = tf.get_variable('w4', shape=[3])

z1 = 3 * w1 + 2 * w2+ w3
z2 = -1 * w3 + w4

""" 前面是ys, 后面是xs """
# grads = tf.gradients([z1, z2], [w1, w2, w3, w4], grad_ys=[[-2.0, -3.0, -4.0], [-2.0, -3.0, -4.0]])
grads = tf.gradients([z1, z2], [w1, w2, w3, w4])

"""
    grade1 = z1/w1 + z2/w2
"""

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(grads))
