import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


"""
    要点1.记忆
        记忆单元存储s,op_fuwu_order.txt,r,s_
        当达到一定步骤数后，学习这些记忆
    要点2.学习
        2.1.更新神经网络
            每一次学习都更新eval_net
            学习步数达到一定次数，才更新target_net
        2.2.
            记忆单元抽样（不用学习全部的记忆）
            计算q_next, q_eval
            由q_next，q_eval计算得到q_target（可以理解为Y，即监督学习中的训练样本）
            Y = WX + B，此处Y对应q_target， X对应s(s是一個1*2的矩阵，二维坐标点嘛)
                这一系列的变换就是矩阵的变化，先映射到高纬度，在映射回来n_action
            q_next, q_eval在这里是一個n * 4的矩阵，表示各个动作对应的值

        2.3 q_target计算
            为了实现有效的反向传递（提升神经网络参数），需要让q现实和q估计的action位置相同
            计算出的q_target作为有效的q现实

        2.4 参数提升(eval_net)
            q_target和q_eval做差得loss，并反向传递即可

    要点3. 消除相关性
        3.1 expression replay
            使用记忆库，从记忆库中随机选取一些记忆，随机抽取这种做法打乱了经历之间的相关性, 也使得神经网络更新更有效率
        3.2 fixed q_targets
            为什么使用两个神经网络，而不是一个。
            一个神经网络的更新频率可能是1000次学习更新一次，另一个是每次都更新一次
            在 DQN 中使用到两个结构相同但参数不同的神经网络, 预测 Q 估计 的神经网络具备最新的参数,
            而预测 Q 现实 的神经网络使用的参数则是很久以前的

"""


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=300, memory_size=500, batch_size=32, e_greedy_increment=None,
                 output_graph=False, ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        """ epsilon最大值 """
        self.epsilon_max = e_greedy
        """ 更换target_net的步数 """
        self.replace_target_iter = replace_target_iter
        """ 记忆上限 """
        self.memory_size = memory_size
        """ 每次更新从记忆中取多少数据 """
        self.batch_size = batch_size
        """ epsilon的增量 """
        self.epsilon_increment = e_greedy_increment
        """ 是否开启探索模式，epsilon有增量则开启探索，没有增量则直接去最大值 """
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, op_fuwu_order.txt, r, s_]
        """ 记忆 """
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        # 输出tensorboard文件
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        """ 记录cost，最后plot出来 """
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            # config of layers
            """ 定义collection的name,后面通过这个name来取值, 记录的值是W 和 B """
            c_names, n_l1, w_initializer, b_initializer = ['eval_net_params',
                                                           tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                                                          tf.random_normal_initializer(0., 0.3),\
                                                          tf.constant_initializer(0.1)

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                """ 只输入状态值 """
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        """ 求误差"""
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        """ 优化，梯度下降 """
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            """ 定义collection的name,后面通过这个name来取值, 记录的值是W 和 B """
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                """ 输入下一个状态，计算的结果是q现实 """
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))
        print(transition)
        quit()

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check if to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        """ 从大量的记忆中，选择one batch的记忆即可以了 """
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # fixed params
        # newest params
        """ 计算q现实,q_next, q估计q_eval, 结果是所有动作的值矩阵 """
        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],
                                       feed_dict={self.s_: batch_memory[:, -self.n_features:],
                                                  self.s: batch_memory[:, :self.n_features], })

        """

            q_target的计算的复杂的原因在于：为了实现有效的反向传递，所以，
            需要q_target的action对应位置和q_eval的对应位置是一致的。

             q_next, q_eval 包含所有 action 的值,我们需要的只是已经选择好的 action 的值, 其他的并不需要
             所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据

             这是我们最终要达到的样子, 比如 q_target - q_eval = [1, 0, 0] - [-1, 0, 0] = [2, 0, 0]
                q_eval = [-1, 0, 0] 表示这一个记忆中有我选用过 action 0,
                而 action 0 带来的 Q(s, a0) = -1, 所以其他的 Q(s, a1) = Q(s, a2) = 0.

                q_target = [1, 0, 0] 表示这个记忆中的 r+gamma*maxQ(s_) = 1,
                而且不管在 s_ 上我们取了哪个 action, 我们都需要对应上 q_eval 中的 action 位置,
                所以就将 1 放在了 action 0 的位置

            下面也是为了达到上面说的目的, 不过为了更方面让程序运算, 达到目的的过程有点不同
            是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
            不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
            使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子
        """
        # change q_target w.r.t q_eval's action
        """ 与q_learning一致，本质还是修改当前状态的动作值，即q_target """
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then back propagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        """ 修正q估计的神经网络，训练的目的是为了让神经网络计算出每一个状态下的各个动作对应的概率值，
            这和q_table来存储的结果是一致的。相当于使用神经来替换q_table这张表.
            这样的好处在于神经网络可以处理趋向于无限的维度(换句话说可以处理极多的状态，极多的动作)
         """
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features], self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
