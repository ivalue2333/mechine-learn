import numpy as np
import pandas as pd
import time

"""
    1. 预设值
"""

N_STATES = 6  # 1维世界的宽度
ACTIONS = ['left', 'right']  # 探索者的可用动作


""" 核心参数 """
# 贪婪度 greedy， 本次我有多大的概率按照最优路线走
EPSILON = 0.9
# 学习率，本次的优化，我要学习多少
ALPHA = 0.1
# 奖励递减值， 步骤越少，说明走的弯路越少，得到的奖励也会也多
GAMMA = 0.9

MAX_EPISODES = 13  # 最大回合数
FRESH_TIME = 0.3  # 移动间隔时间

"""
    2. 定义q 表 q_table
"""


def build_q_table(n_states, actions):
    # q_table 全 0 初始
    # columns 对应的是行为名称
    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions, )
    return table


"""
    3. 选择行为, 在某个 state 地点
"""


def choose_action(state, q_table):
    # 选出这个 state 的所有 action 值
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        # 非贪婪 or 或者这个 state 还没有探索过
        action_name = np.random.choice(ACTIONS)
    else:
        # 贪婪模式
        action_name = state_actions.argmax()
    return action_name


"""
    4. 环境反馈
"""


def get_env_feedback(S, A):
    """
    返回上个 state 和 action 作用下的 R(reward)
    返回下一个 state (S_)
    :param S:
    :param A:
    :return: R , S_
    """
    # This is how agent will interact with the environment
    if A == 'right':
        # move right
        # terminate
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0

    else:
        # move left
        R = 0
        if S == 0:
            # reach the wall
            S_ = S
        else:
            S_ = S - 1
    return S_, R


"""
    5. 环境更新
"""


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            # take action & get next state and reward
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                # next state is not terminal

                # 这一行的最大 action 概率,再做点运算，得到Q现实
                # gamma学习奖励递减值，意味着如果学习使用的步骤太多，那么q_target变小（说明你的学习走了弯路了）
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                # next state is terminal
                q_target = R
                # terminate this episode
                is_terminated = True

            # update
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            # move to next state
            S = S_

            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
