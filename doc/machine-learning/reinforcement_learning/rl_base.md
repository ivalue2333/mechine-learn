# 强化学习

[TOC]

## Synopsis

## Refers

莫凡的教程：https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/1-1-B-RL-methods/

深度强化学习劝退文：https://www.leiphone.com/news/201802/kySbslzMWzUXIAbt.html

policy gradient：https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf

大神的解释：http://karpathy.github.io/2016/05/31/rl/

http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/pg.pdf

## Details

### 简单分类

#### model-free和model-based

#### 基于概率和基于价值

#### 回合更新和单步更新

#### 在线学习和离线分析

### Q_learning

#### 重点

q_估计对应的是当前状态，q_现实对应的是下一个状态



q_learning的核心就是一张q表(q_table)，它表示了在每一个状态，执行各种可能的动作的概率

训练过程中通过目标激励(reward)，不断使得q_table各个状态下的执行的动作是，向着目标前进的。

#### steps

1. 选择动作
2. 在环境中模拟动作，并获得该动作下的 下一个状态和奖励
3. 学习这个状态和奖励，也就意味着更新q表
4. 更新当前状态为刚才模拟执行动作的返回状态（其实相当于执行了这个动作，刚才只是模拟了这个动作）
5. 又选择动作...开始循环了

### Sarsa

state-action-reward-state\_-action\_

1. 选择动作
2. 环境中模拟： Q(s, a)这个状态机下，得到P(s_, r)，s表示状态，r表示奖励
3. 选择下一个动作：在s\_的基础上查q表得到 a\_
4. 学习（更新Q表）：以s, a, r, s\_, a\_ 为参数，这里q估计就是去下一个状态的概率[a\_, s\_]，而q_learning 是去[a\_,: ]中的最大值。
5. a, s = a\_, s\_， 又开始学习

### Sarsa-lambda

Sarsa-lambda 是基于 Sarsa 方法的升级版, 他能更有效率地学习到怎么样获得好的 reward. 如果说 Sarsa 和 Qlearning 都是每次获取到 reward, 只更新获取到 reward 的前一步. 那 Sarsa-lambda 就是更新获取到 reward 的前 lambda 步. lambda 是在 [0, 1] 之间取值,

如果 lambda = 0, Sarsa-lambda 就是 Sarsa, 只更新获取到 reward 前经历的最后一步.

如果 lambda = 1, Sarsa-lambda 更新的是 获取到 reward 前所有经历的步.

### DQN

我们使用表格来存储每一个状态 state, 和在这个 state 每个行为 action 所拥有的 Q 值. 而当今问题是在太复杂, 状态可以多到比天上的星星还多(比如下围棋). 如果全用表格来存储它们, 恐怕我们的计算机有再大的内存都不够, 而且每次在这么大的表格中搜索对应的状态也是一件很耗时的事。不过, 在机器学习中, 有一种方法对这种事情很在行, 那就是神经网络. 我们可以将状态和动作当成神经网络的输入, 然后经过神经网络分析后得到动作的 Q 值, 这样我们就没必要在表格中记录 Q 值, 而是直接使用神经网络生成 Q 值. 还有一种形式的是这样, 我们也能只输入状态值, 输出所有的动作值, 然后按照 Q learning 的原则, 直接选择拥有最大值的动作当做下一步要做的动作. 我们可以想象, 神经网络接受外部的信息, 相当于眼睛鼻子耳朵收集信息, 然后通过大脑加工输出每种动作的值, 最后通过强化学习的方式选择动作.

总结一下：

- 将状态和动作作为神经网络的输入，通过神经网络分析后就可以得到动作的Q值，这样不用存储Q值，因为映射逻辑在神经网络中。
- 将状态作为神经网络的输入，输出所有的动作，直接选择有最大值的动作

DQN 的精髓部分之一: 记录下所有经历过的步, 这些步可以进行反复的学习, 所以是一种 off-policy 方法, 你甚至可以自己玩, 然后记录下自己玩的经历, 让这个 DQN 学习你是如何通关的.

target_net 用于预测 q_target 值, 他不会及时更新参数. eval_net 用于预测 q_eval, 这个神经网络拥有最新的神经网络参数。 target_net 是 eval_net 的一个历史版本, 拥有 eval_net 很久之前的一组参数, 而且这组参数被固定一段时间, 然后再被 eval_net 的新参数所替换. 而 eval_net 是不断在被提升的, 所以是一个可以被训练的网络 trainable=True. 而 target_net 的 trainable=False.

1. 建立初始化参数
2. 选行为
3. 存储过渡态(s, a, r, s_)（状态，动作，回报，下一个状态）
4. 经过两百步学习一次
   1. 经过replace_target_iter（300）次更新一次target_net
      1. 将最新的eval_net赋值给target_net

### doule dqn

主要解决dqn中的过估计(overestimate)的问题

没太懂，不过传统的dqn的估计其q_eval确实会超过实际的最大值

### policy-gradient

直接输出动作，可以在连续区间选择动作。

输出动作，反向传递的就是让这个动作在下一次更有可能发生。那么怎样判断这个动作是不是应该更又可能发生呢？reward，奖励信息来告诉神经网络这个动作的概率增加幅度0.1 和10显然是一个动作更不应该发生，另一个动作更应该发生。

1. 初始化参数
2. 建立policy gradient神经网络
3. 选择行为
4. 存储回合transaction
5. 学习更新参数

### actor-critic

解决policy gradient，但是引入了行为的相关性的问题，所以这个模型的结果不好。

Actor-Critic 的 Actor 的前生是 Policy Gradients, 这能让它毫不费力地在连续动作中选取合适的动作, 而 Q-learning 做这件事会瘫痪. 那为什么不直接用 Policy Gradients 呢? 原来 Actor Critic 中的 Critic 的前生是 Q-learning 或者其他的 以值为基础的学习法 , 能进行单步更新, 而传统的 Policy Gradients 则是回合更新, 这降低了学习效率.

结合了 Policy Gradient (Actor) 和 Function Approximation (Critic) 的方法. Actor 基于概率选行为, Critic 基于 Actor 的行为评判行为的得分, Actor 根据 Critic 的评分修改选行为的概率.简单说，Actor 要朝着更有可能获取大 Q 的方向修改动作参数了.

总结来说就是

- 可以选择连续的动作
- 可以单步更新，而不用回合更新，提高学习效率

Actor-Critic 涉及到了两个神经网络, 而且每次都是在连续状态中更新参数, 每次参数更新前后都存在相关性, 导致神经网络只能片面的看待问题, 甚至导致神经网络学不到东西. Google DeepMind 为了解决这个问题, 修改了 Actor Critic 的算法。(ddqg)

1. 建立critic神经网络
2. 建立actor神经网络
3. actor选择动作
4. critic 学习actor动作a产生的(s, r, s_)并返回得分
5. actor学习critic的打分(s, a, td_error)
   1. 依据当前动作a, 和神经网络的返回值acts_prob

### Deep Deterministic Policy Gradient

> ddpg  = actor-critic + dqn

一句话概括 DDPG: Google DeepMind 提出的一种使用 Actor Critic 结构, 但是输出的不是行为的概率, 而是具体的行为, 用于连续动作 (continuous action) 的预测. DDPG 结合了之前获得成功的 DQN 结构, 提高了 Actor Critic 的稳定性和收敛性.



