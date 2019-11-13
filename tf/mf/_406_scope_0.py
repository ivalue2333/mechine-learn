import tensorflow as tf


"""
理解：
    tf.get_variable: 配合tf.variable_scope 及 scope.reuse_variables() 使得 var1 和 var12是同一个变量（name相同）
    tf.Variable: var2, var21, var22是三个不同的变量

    所以如果要复用变量,使用 tf.get_variable
"""

# name_scope
# with tf.name_scope("a_name_scope"):
#     initializer = tf.constant_initializer(value=1)
#     var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
#     var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
#     var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
#     var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)


# variable_scope
with tf.variable_scope("a_variable_scope") as scope:
    initializer = tf.global_variables_initializer(value=1)
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
    scope.reuse_variables()
    var12 = tf.get_variable(name='var1')
    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
    var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(var1.name)        # var1:0
    print(sess.run(var1))   # [ 1.]
    print('-'*10)
    print(var12.name)        # var1:0
    print(sess.run(var12))   # [ 1.]
    print('-'*10)
    print(var2.name)        # a_name_scope/var2:0
    print(sess.run(var2))   # [ 2.]
    print('-' * 10)
    print(var21.name)       # a_name_scope/var2_1:0
    print(sess.run(var21))  # [ 2.0999999]
    print('-' * 10)
    print(var22.name)       # a_name_scope/var2_2:0
    print(sess.run(var22))  # [ 2.20000005]
