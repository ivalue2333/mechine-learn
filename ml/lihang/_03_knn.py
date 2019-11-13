import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


"""
    线性扫描的knn没有训练过程
    输入一个点A，计算所有train_data和点A的距离，其中和点A最近的k个点构成一个集合，在这个集合中，
    获得里面的点B1, B2...Bk的label，哪个label数量最多，这个点A就属于哪个label

    kd-tree 没有搞了
"""


def L(x, y, p=2):
    """
    p = 1 曼哈顿距离
    p = 2 欧氏距离
    :param x:
    :param y:
    :param p:
    :return:
    """
    # x1 = [1, 1], x2 = [5,1]
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1/p)
    else:
        return 0


class KnnModel:
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        """
        parameter: n_neighbors 临近点个数
        parameter: p 距离度量
        """
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        # 取出n个点
        knn_list = []
        for i in range(self.n):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))
        for i in range(self.n, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])
        # 统计
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)
        max_count = sorted(count_pairs, key=lambda x: x)[-1]
        return max_count

    def score(self, X_test, y_test):
        right_count = 0
        n = 10
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


my_model = False

if my_model:
    model = KnnModel(X_train, y_train)
    print(model.score(X_test, y_test))

    test_point = np.array([6.0, 3.0])
    test_label = model.predict(test_point)
    test_color = 'g' if test_label == 0 else 'b'

    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], color='g', label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], color='b', label='1')
    plt.plot(test_point[0], test_point[1], color=test_color, marker='x', label='test_point')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()
else:
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    test_point = np.array([6.0, 3.0])
    print(model.predict(test_point.reshape(1, -1)))