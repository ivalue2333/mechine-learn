import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # op_fuwu_order.txt list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    """
        1. q_predict，q估计，q估计为当前状态，当前动作对应的值
            （本次学习就是在让这个值变大（当前状态下选择此动作正确性高），变小（当前状态下次动作正确性低））
        2. q_target，q现实，（现实你获得的值是） r + gamma * max(q(s_))
            即回报 + 递降奖励因子 * 下一个状态行最大的值

        3. q[s, op_fuwu_order.txt] = lr * (q_target -q_predict)
            即更新当前的状态和行动，对应的值为 学习效率 * （q现实 - q估计），（可以忽略学习效率来理解）

        显然，回报越大，下一个动作的最大值(概率)越大，说明这个行动越正确。对应的 q_target的值也越大。
        这并不是严格意义上的推导，只是直观的理解。
    """
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state, ))
