import tensorflow as tf

"""
    ((3-1)**2 + (3-2)**2 + (3-3)**2) / 3
    差的平方 / n
"""

# q_target = [[3,3],[3,3]]
# q_eval = [[1,3], [1,2]]

q_target = [3,3,3]
q_eval = [1,2,3]

td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q_eval)
with tf.Session() as sess:
    a = sess.run(td_error)
    print(a)