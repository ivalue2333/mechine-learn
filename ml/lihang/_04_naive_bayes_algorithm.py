import numpy as np
import math
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
# 高斯模型，伯努利模型，多项式模型
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    return data[:, :-1], data[:, -1]


class NaiveBayesModel:
    """
    通过计算各个feature的mead, std，并将这个feature认为是一個正太分布，得到正太分布的概率密度函数
    """
    def __init__(self):
        self.model = dict()

    @staticmethod
    def compute(xs):
        summaries = [(np.mean(x_col), np.std(x_col)) for x_col in zip(*xs)]
        return summaries

    @staticmethod
    def gaussian_probability(x, mean, std):
        exponent = math.exp(-math.pow((x-mean), 2) / (2 * math.pow(std, 2)))
        return math.pow(math.sqrt(2 * math.pi) * std, -1) * exponent

    def fit(self, xs, ys):
        labels = list(set(ys))
        label_x_dict = {label: [] for label in labels}
        for x, y in zip(xs, ys):
            label_x_dict[y].append(x)

        # 在这种假设下，相当于计算除了 P(xj / y = ck) = f(xj / y=ck)这是个条件概率密度函数
        # 每一个y，这里都有对应的密度函数，给一个x, 那么就能计算出概率
        self.model = {y: self.compute(xs) for y, xs in label_x_dict.items()}

    def predict(self, xs):
        label_prob_dict = {}
        for label, values in self.model.items():
            probability = 1
            for i in range(len(xs)):
                probability *= self.gaussian_probability(xs[i], values[i][0], values[i][1])
            label_prob_dict[label] = probability
        return sorted(label_prob_dict.items(), key=lambda item: item[1])[-1][0]

    def score(self, xs, ys):
        right_count = 0
        for x, y in zip(xs, ys):
            label = self.predict(x)
            if label == y:
                right_count += 1
        return float(right_count) / len(ys)


X, Y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

print('一个数据: ', X_test[0], y_test[0])

my_model = True

if my_model:
    model = NaiveBayesModel()
    model.fit(X_train, y_train)
    print('score: ', model.score(X_test, y_test))
    print('predict: ', model.predict([4.4, 3.2, 1.3, 0.2]))

else:
    # model = GaussianNB()
    model = BernoulliNB()
    model.fit(X_train, y_train)
    print('score: ', model.score(X_test, y_test))
    test_data = np.array([4.4, 3.2, 1.3, 0.2]).reshape(1, -1)
    print('test_data: ', test_data)
    print('predict: ', model.predict(test_data))