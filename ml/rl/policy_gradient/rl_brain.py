import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


"""

使用 softmax 和神经网络的最后一层 logits 输出和真实标签 (self.tf_acts) 对比的误差.
并将神经网络的参数按照这个真实标签改进. 这显然和一个分类问题没有太多区别.
我们能将这个 neg_log_prob 理解成 cross-entropy 的分类误差.
分类问题中的标签是真实 x 对应的 y, 而我们 Policy gradient 中, x 是 state, y 就是它按照这个 x 所做的动作号码.
所以也可以理解成, 它按照 x 做的动作永远是对的 (出来的动作永远是正确标签),
它也永远会按照这个 “正确标签” 修改自己的参数. 可是事实却不是这样, 他的动作不一定都是 “正确标签”,
这就是强化学习(Policy gradient)和监督学习(classification)的不同

    再接着上面强调一下，监督学习的正确标签就是正确的，强化学习的的“正确标签”只是在学习到那一刻认为是正确的

    为了确保这个动作真的是 “正确标签”, 我们的 loss 在原本的 cross-entropy 形式上乘以 vt,
用 vt 来告诉这个 cross-entropy 算出来的梯度是不是一个值得信任的梯度. 如果 vt 小, 或者是负的,
就说明这个梯度下降是一个错误的方向, 我们应该向着另一个方向更新参数, 如果这个 vt 是正的, 或很大,
vt 就会称赞 cross-entropy 出来的梯度, 并朝着这个方向梯度下降
"""


class PolicyGradient:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95, output_graph=False, ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        """ dense全连接层 """
        layer = tf.layers.dense(inputs=self.tf_obs, units=10, activation=tf.nn.tanh,  # tanh activation
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                bias_initializer=tf.constant_initializer(0.1), name='fc1')
        # fc2
        all_act = tf.layers.dense(inputs=layer, units=self.n_actions, activation=None,
                                  kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                  bias_initializer=tf.constant_initializer(0.1), name='fc2')

        """ 将动作的值装换为概率, TOKNOWN 动作的值是怎么转为概率的呢 """
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            """ 最大化reward(log_p * R) 和 最小化 -(log_p * R)等价，因为tf只有最小化函数，没有最大化函数 """
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # this is negative log of chosen action
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,
                                                                          labels=self.tf_acts)
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        """ prob_weights就是概率了 """
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob

        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        # self.tf_obs, shape=[None, n_obs]
        # self.tf_acts, shape=[None, ]
        # self.tf_vt, shape=[None, ]
        self.sess.run(self.train_op, feed_dict={self.tf_obs: np.vstack(self.ep_obs), self.tf_acts: np.array(self.ep_as),
                                                self.tf_vt: discounted_ep_rs_norm, })

        """ 一个回合玩完了，就清空列表(observation, action, reward) """
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
