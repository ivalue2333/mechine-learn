"""
理解：
    RNN（Recurrent Neural Network）
    如何让数据间的关联也被 NN 加以分析呢? 想想我们人类是怎么分析各种事物的关联吧,
    最基本的方式,就是记住之前发生的事情. 那我们让神经网络也具备这种记住之前发生的事的能力.
    在分析 Data0 的时候, 我们把分析结果存入记忆. 然后当分析 data1的时候, NN会产生新的记忆,
    但是新记忆和老记忆是没有联系的. 我们就简单的把老记忆调用过来, 一起分析. 如果继续分析更多的有序数据 ,
    RNN就会把之前的记忆都累积起来, 一起分析.

    梯度消失
        他在每一步都会 乘以一个自己的参数 W. 如果这个 W 是一个小于1 的数,
        比如0.9. 这个0.9 不断乘以误差, 误差传到初始时间点也会是一个接近于零的数,
        所以对于初始时刻, 误差相当于就消失了
    梯度爆炸
         反之如果 W 是一个大于1 的数, 比如1.1 不断累乘, 则到最后变成了无穷大的数,
         RNN被这无穷大的数撑死了, 这种情况我们叫做剃度爆炸, Gradient exploding.
         这就是普通 RNN 没有办法回忆起久远记忆的原因

    LSTM
    长短期记忆（Long short-term memory, LSTM）是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。
    简单来说，就是相比普通的RNN，LSTM能够在更长的序列中有更好的表现。

    LSTM 他多了一个 控制全局的记忆, 我们用粗线代替. 为了方便理解, 我们把粗线想象成电影或游戏当中的 主线剧情.
    而原本的 RNN 体系就是 分线剧情。
    三个控制器都是在原始的 RNN 体系上

    输入方面, 如果此时的分线剧情对于剧终结果十分重要, 输入控制就会将这个分线剧情按重要程度 写入主线剧情 进行分析
    忘记方面, 如果此时的分线剧情更改了我们对之前剧情的想法,
        那么忘记控制就会将之前的某些主线剧情忘记, 按比例替换成现在的新剧情
        所以 主线剧情的更新就取决于输入和忘记控制
    输出方面, 输出控制会基于目前的主线剧情和分线剧情判断要输出的到底是什么
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)   # set random seed

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001                  # learning rate
training_iter = 100000     # train step 上限
batch_size = 128
n_steps = 28                # time steps


n_inputs = 28               # MNIST data input (img shape: 28*28)
n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              # MNIST classes (0-9 digits)

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def rnn(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell

    # basic LSTM Cell.
    if int(tf.__version__.split('.')[1]) < 12 and int(tf.__version__.split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # unpack to list [(batch, outputs)..] * steps
    if int(tf.__version__.split('.')[1]) < 12 and int(tf.__version__.split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))               # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']        # shape = (128, 10)

    return results


# 定义训练张量
prediction = rnn(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# 计算准确率的张量
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    if int(tf.__version__.split('.')[1]) < 12 and int(tf.__version__.split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iter:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
        step += 1
