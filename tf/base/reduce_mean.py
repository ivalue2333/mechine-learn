import tensorflow as tf
import numpy as np

""" 计算平均值 """

a = np.asarray([1, 2, 3, 4, 5, 6])
a = a.reshape((2, 3))

print(a)
print('*' * 10)

# 可以直接转换numpy数组ndarray
# op_fuwu_order.txt = [[1,2,3], [4,5,6]]
x = tf.cast(a, tf.float32)
ops_1 = tf.reduce_mean(x)
ops_2 = tf.reduce_mean(x, axis=0)
ops_3 = tf.reduce_mean(x, axis=1)
with tf.Session() as sess:
    ops_res_1 = sess.run(ops_1)
    ops_res_2 = sess.run(ops_2)
    ops_res_3 = sess.run(ops_3)
print(ops_res_1)
print(ops_res_2)
print(ops_res_3)

