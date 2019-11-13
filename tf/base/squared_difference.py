import tensorflow as tf
import numpy as np

""" 差的平方 """

a = np.asarray([1, 2, 3, 4, 5, 6])
a = a.reshape((2, 3))

print(a)
print('*' * 10)

# 可以直接转换numpy数组ndarray
# op_fuwu_order.txt = [[1,2,3], [4,5,6]]
x = tf.cast(a, tf.float32)
b = [[2,3,4], [1,2,3]]
""" 下面两种都可以 """
# y = tf.cast(b, tf.float32)
y = tf.constant(b, dtype=tf.float32)
ops_1 = tf.squared_difference(x, y)
with tf.Session() as sess:
    ops_res_1 = sess.run(ops_1)
print(ops_res_1)
