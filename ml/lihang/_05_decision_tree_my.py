from math import log

import numpy as np
import pandas as pd


def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels


class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {'label:': self.label, 'feature': self.feature, 'tree': self.tree}

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)


class Id3DecisionTreeModel:

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = Node()

    def calc_ent(self, datasets):
        """
        计算经验熵
        :param datasets:
        :return:
        """
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(ck / data_length) * log(ck / data_length, 2) for ck in label_count.values()])
        return ent

    def calc_cond_ent(self, datasets, axis=0):
        """
        计算经验条件熵
        :param datasets: 数据集
        :param axis: 表示取第几个feature
        :return:
        """
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(dj) / data_length) * self.calc_ent(dj) for dj in feature_sets.values()])
        return cond_ent

    def info_gain(self, datasets, axis):
        """ 计算信息增益 """
        return self.calc_ent(datasets) - self.calc_cond_ent(datasets, axis)

    def info_gain_train(self, datasets):
        """  """
        best_feature = []
        for c in range(len(datasets[0]) - 1):
            c_info_gain = self.info_gain(datasets, axis=c)
            best_feature.append((c, c_info_gain))
        best_ = max(best_feature, key=lambda x: x[-1])
        return best_

    def train(self, train_data):
        """
        递归的生成id3的决策树
        :param train_data:
        :return:
        """
        # features, 存放feature的数组
        y_train, features = train_data.iloc[:, -1], train_data.columns[:-1]
        # 1,若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if len(y_train.value_counts()) == 1:
            return Node(root=True, label=y_train.iloc[0])

        # 2, 若A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 3,计算最大信息增益 同5.1,Ag为信息增益最大的特征
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]

        # 4,Ag的信息增益小于阈值eta,则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if max_info_gain < self.epsilon:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 5,构建Ag子集
        node_tree = Node(root=False, feature_name=max_feature_name, feature=max_feature)

        """ 获取当前最大信息增益的那个feature，所对应的所有值 """
        feature_list = train_data[max_feature_name].value_counts().index

        for f in feature_list:
            """ sub_train_df是这样一个df, 它是train_data中feature_name...  """
            """ 比如它是feature_name是有自己的房子，f 依次是是，否， 然后例如取所有有房子的数据，然后取到的数据中，将有房子这一列去除 """
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)
            # 6, 递归生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)
        return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)

ds, ls = create_data()
train_data = pd.DataFrame(ds, columns=ls)

model = Id3DecisionTreeModel()

model.fit(train_data)

print('labels: ', ls)
print(model.predict(['老年', '否', '是', '一般']))
