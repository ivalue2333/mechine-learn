# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_csv(file_path, seq):
    """
    以\t作为间隔
    :param file_path:
    :return:
    """
    df = pd.read_csv(file_path,  sep=seq, header=0, na_filter=False)
    return np.asarray(df, dtype=np.float)


def compute_edist(arr_a, arr_b):
    """
    计算欧式距离
    :param arr_a: np.array一维数组
    :param arr_b: np.array一维数组
    :return:
    """
    return np.math.sqrt(sum(np.power(arr_a - arr_b, 2)))


class KMeansClassifier:
    def __init__(self, k, ndarray, max_iter=500):
        # 存放最小索引和距离（op_fuwu_order.txt, b）
        self.cluster = None
        # 存放质心
        self.centers = None
        self.ndarray = ndarray
        self.k = k
        self.max_iter = max_iter

    def random_center(self):
        # 数据集的维度
        n = self.ndarray.shape[1]
        # k 个 n 维 质心 => k 行 n 列 数据
        center_array = np.empty([self.k, n])
        for j in range(n):
            # 第j列全部数据
            min_j = min(self.ndarray[:, j])
            range_j = max(self.ndarray[:, j]) - min_j

            # num_j = (min_j + range_j * np.random.rand(self.k, 1)).flatten()
            num_j = (min_j + range_j * np.random.rand(self.k))
            center_array[:, j] = num_j
        self.centers = center_array

    def fit(self):
        # 样本行数
        m = self.ndarray.shape[0]
        self.cluster = np.zeros((m, 2))

        for _ in range(self.max_iter):
            cluster_changed = False
            # 遍历所有的点，在k个质心中为他们找新的聚类
            for i in range(m):
                # 最小距离先设置为inf
                min_dist = np.inf
                # 归属哪个质心先设置为-1
                min_index = -1
                arr_a = self.ndarray[i]
                for j in range(self.k):
                    center = self.centers[j]
                    edist = compute_edist(arr_a, center)
                    if min_dist > edist:
                        min_dist = edist
                        min_index = j

                if self.cluster[i, 0] != min_index or self.cluster[i, 1] > min_dist**2:
                    self.cluster[i] = min_index, min_dist**2
                    cluster_changed = True
            if not cluster_changed:
                break

            # 重新计算质心
            for i in range(self.k):
                # bad way
                # index_all = self.cluster[..., 0]
                # value = np.nonzero(i == index_all)
                # better way
                value = np.where(self.cluster[..., 0] == i)
                points_in_cluster = self.ndarray[value[0]]

                imeans = np.mean(points_in_cluster, axis=0)
                self.centers[i] = imeans

    def plt(self):
        colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y']

        for i in range(self.k):
            indexes = np.where(self.cluster[..., 0] == i)
            x = self.ndarray[indexes[0], 0]
            y = self.ndarray[indexes[0], 1]

            num_label = str(i)
            for j in range(len(x)):
                plt.text(x[j], y[j], num_label, color=colors[i], fontdict={'weight': 'bold', 'size': 6})
            plt.scatter(self.centers[i, 0], self.centers[i, 1], marker='x', color=colors[i], linewidths=7)
        plt.axis([-7, 7, -7, 7])
        plt.show()

if __name__ == '__main__':
    nd = load_csv('data/testSet.txt', '\t')
    # 三个质心
    kmeans = KMeansClassifier(3, nd)
    kmeans.random_center()
    kmeans.fit()
    kmeans.plt()