import tensorflow as tf


"""
    collection相当于一个容器
    add_to_collection相当于向容器添加张量
    get_collection相当于从指定名称的容器中获取张量
"""


v1 = tf.get_variable(name='v1', shape=[1], initializer=tf.constant_initializer(1))
tf.add_to_collection('loss', v1)
v2 = tf.get_variable(name='v2', shape=[1], initializer=tf.constant_initializer(2))
tf.add_to_collection('loss', v2)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    """ 获取的是张量 """
    print(tf.get_collection('loss'))
    """ 结果是一个数组 """
    print(sess.run(tf.get_collection('loss')))
    """ add_n 将这个list的数据做累加 """
    print(sess.run(tf.add_n(tf.get_collection('loss'))))
