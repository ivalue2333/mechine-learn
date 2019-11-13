import tensorflow as tf

"""
    assign: 将值复制给一个tensor变量
"""


a = tf.Variable([1,2,3])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(tf.assign(a, [1,2,4])))
