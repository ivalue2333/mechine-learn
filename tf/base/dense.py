import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

"""
    使用tf的全连接层，来训练简单抛物线回归

    dense参数
        inputs: 输入数据，2维tensor.
        units: 该层的神经单元结点数。dimensionality of the output space.
        activation: 激活函数.
        kernel_initializer: 卷积核的初始化器.
        bias_initializer: 偏置项的初始化器，默认初始化为0.
        kernel_regularizer: 卷积核化的正则化，可选.
        bias_regularizer: 偏置项的正则化，可选.
        activity_regularizer: 输出的正则化函数.
        trainable: Boolean型，表明该层的参数是否参与训练。如果为真则变量加入到图集合中
        name: 层的名字.

    这里没有数据，所以fake了一些数据，将这些数据看做是前期收集的数据就可以了
    核心的方程式 Y = WX + B（线性代数）
    在这个训练中，
        1. 投递X数据，得到Y_pred（Y预测）
        2. 计算loss，f(Y - Y_pred)
        3. 生成优化器(设定特定的学习效率)
        4. 通过优化器，反向传递loss，修正隐藏层(dense)的 W 和 B

"""


tf.set_random_seed(1)
np.random.seed(1)

# fake data
x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise                          # shape (100, 1) + some noise

# 不加噪声
# y = np.power(x, 2)                          # shape (100, 1) + some noise


# plot data
plt.scatter(x, y)
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)     # input x
tf_y = tf.placeholder(tf.float32, y.shape)     # input y

# neural network layers
""" 全连接层 """
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)          # hidden layer
output = tf.layers.dense(l1, 1)                     # output layer

loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph

plt.ion()   # something about plotting

for step in range(100):
    # train and net output
    _, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
    if step % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

print(pred)
plt.ioff()
plt.show()