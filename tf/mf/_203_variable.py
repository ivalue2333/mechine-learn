import tensorflow as tf

"""
理解：
    变量有点像是一个"算子"，tf中的变量其实不应该理解为值，应该理解为"算子"，只有在session的计算中才能得到值
"""

state = tf.Variable(0, name='counter')

# 定义常量 one
one = tf.constant(1)

# 定义加法步骤 (注: 此步并没有直接计算， 这意味着new_value不是一个值，而是一个算子)
new_value = tf.add(state, one)

# 将 State 更新成 new_value
update = tf.assign(state, new_value)

# 如果定义 Variable, 就一定要 initialize
init = tf.global_variables_initializer()  # 替换成这样就好

# 使用 Session
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
        # print(state)

# 直接 print(state) 不起作用！！
# 一定要把 sess 的指针指向 state 再进行 print 才能得到想要的结果！
