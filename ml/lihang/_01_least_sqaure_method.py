import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq


"""
    leastsq：最小二乘法，这里使用最小二乘法做参数拟合
"""


# 随机数种子...
np.random.seed(1)


class SimpleModel:
    def __init__(self, m=3):
        self.M = m
        self.ws = np.random.rand(self.M + 1)

        self.regularization = 0.0001

    def fit(self, xs, ys, regular=False):
        ws_init = self.ws
        if not regular:
            self.ws = leastsq(self.train_loss, ws_init, args=(xs, ys))[0]
        else:
            # 加上ws的惩罚因子，正则化
            self.ws = leastsq(self.train_loss_regularization, ws_init, args=(xs, ys))[0]

    def predict(self, xs, ws=None):
        if ws is None:
            ws = self.ws
        poly_func = np.poly1d(ws)
        return poly_func(xs)

    def train_loss(self, ws_train, xs, ys):
        ret = self.predict(xs, ws_train) - ys
        return ret

    def train_loss_regularization(self, ws_train, xs, ys):
        ret = self.predict(xs, ws_train) - ys
        ret = np.append(ret, np.sqrt(0.5 * self.regularization * np.square(ws_train)))
        return ret

    def test_loss(self, xs, ys):
        return self.predict(xs) - ys


# 目标函数
def real_func(x):
    return np.sin(2*np.pi*x)


# 十个点
xs = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 1000)
# 加上正态分布噪音的目标函数的值
ys = real_func(xs) + np.random.normal(0, 0.1, (10, ))

# 尝试 3，6，9...
model = SimpleModel(3)
print('训练前ws', model.ws)
model.fit(xs, ys)
print('训练后ws', model.ws)
print('test loss', np.sqrt(np.sum(np.square(model.test_loss(xs,ys)))))

# 可视化
plt.plot(x_points, real_func(x_points), label='real')
plt.plot(x_points, model.predict(x_points), label='fitted curve')
plt.plot(xs, ys, 'bo', label='noise')
plt.legend()
plt.show()
