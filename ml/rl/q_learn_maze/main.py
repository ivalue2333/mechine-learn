from ml.rl.q_learn_maze.maze_env import Maze
from ml.rl.q_learn_maze.rl_brain import QLearningTable


"""
    As we all know， q_learning 的核心就是 更新 q_表， q_表就是在各个状态，各种行为的概率的集合
"""



def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            """ 学习其实不太关心这一步具体怎么走的
                这一步的核心返回必须包含
                    下一个观察点，回报，是否完成（抵达目的地）
             """
            observation_, reward, done = env.step(action)

            # RL learn from this transition which means update q_table
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()