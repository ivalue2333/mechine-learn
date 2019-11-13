import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
理解：
    1: 每次训练的输入数据是相同的
    2: 每次训练后得到loss， 并通过GradientDescentOptimizer (optimizer) 反向传递，修正Weights 和 biases
    3: 下次训练使用通过optimizer 修正过的Weights 和 biases继续训练，得到新的lose, 并循环到第2步
    4: 模型就是最终的Weights 和 biases，目前还没看到保存模型的地方，还没有看到用模型预测的地方
"""


# 添加层
# weights为一个in_size行, out_size列的随机变量矩阵
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 导入数据
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 利用占位符定义我们所需的神经网络的输入。 tf.placeholder()就是代表占位符，这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 定义隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# 定义输出层
prediction = add_layer(l1, 10, 1, activation_function=None)

# 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

# 减小损失
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 使用变量时，都要对它进行初始化，这是必不可少的。
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 使用plt 可视化
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
# 本次运行请注释，全局运行不要注释
plt.ion()
plt.show()


# 训练
# training
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 10 == 0:
        # to see the step improvement
        sess.run(loss, feed_dict={xs: x_data, ys: y_data})

        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)




