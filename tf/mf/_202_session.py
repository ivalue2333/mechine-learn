import tensorflow as tf


"""
理解:
    session对象可调用"算子"（将算子作为参数传入）， 并返回计算结果
"""

matrix1 = tf.constant([[3, 3]])
print(matrix1)

matrix2 = tf.constant([[2], [2]])
print(matrix2)

product = tf.matmul(matrix1, matrix2)

# method 1
sess = tf.Session()
result1 = sess.run(product)
print(result1)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)