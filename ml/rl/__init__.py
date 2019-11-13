import pandas as pd
import numpy as np


def build_q_table(n_states, actions):
    # q_table 全 0 初始
    # columns 对应的是行为名称
    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions, )
    return table


acs = ['l', 'r']
q = build_q_table(10,acs)
q.iloc[3, 0] = 0.1
q.iloc[3, 1] = 0.9

d1 = q.iloc[3,:]
print(d1)
print(d1.max())

print('-' * 10)


a = [1,2,3,4]
b = [5,6,7,8]
c = [9,10,11,12]
print("op_fuwu_order.txt=",a)
print("b=",b)
print("c=",c)

print("增加一维，新维度的下标为0")
d=np.stack((a,b,c),axis=0)
# print(np.stack((op_fuwu_order.txt, b, c)))
print(d)

print("增加一维，新维度的下标为1")
d=np.stack((a,b,c),axis=1)
print(d)

print('vstack')
print(np.vstack((a,b,c)))
print(np.vstack([1, 2, 3, 4]))



