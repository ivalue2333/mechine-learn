import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron


"""
    李航《统计学习方法》-感知机，简单二元分类
"""


class PerceptronModel:
    def __init__(self):
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1
        # self.data = data

    def func(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    # 随机梯度下降法
    def fit(self, X_train, y_train):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.func(X, self.w, self.b) <= 0:
                    self.w += self.l_rate * np.dot(y, X)
                    self.b += self.l_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True
            print(wrong_count, self.w, self.b)
        return 'Perceptron Model!'

    def score(self):
        pass


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target

# 只取这几个标签的数据
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
df.label.value_counts()

# plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], marker='+', label='0')
# plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend()
# plt.show()

# 第0列，第1列，最后一列
data = np.array(df.iloc[:100, [0, 1, -1]])

# 强行改为 1, -1的二元分类
for x in np.nditer(data[:, -1], op_flags=['readwrite']):
    if x == 0:
        x[...] = -1
# print(data[:, -1])

X, y = data[:,:-1], data[:,-1]


my_model = True


if my_model:
    perceptron = PerceptronModel()
    perceptron.fit(X, y)

    x_points = np.linspace(4, 7, 10)
    # w1x + w2y + b = 0 就是这个超平面（在这里是直线（两个变量））
    y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
    plt.plot(x_points, y_)

    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()

else:
    clf = Perceptron(fit_intercept=False, shuffle=False)
    clf.fit(X, y)
    print(clf.coef_)
    # 截距
    print(clf.intercept_)
    x_points = np.arange(4, 8)
    y_ = -(clf.coef_[0][0] * x_points + clf.intercept_) / clf.coef_[0][1]
    plt.plot(x_points, y_)

    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()