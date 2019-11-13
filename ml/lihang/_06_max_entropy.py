"""
这里主要涉及到了拉格朗日函数。
拉格朗日乘数法（以数学家约瑟夫·拉格朗日命名）是一种寻找多元函数在其变量受到一个或多个条件的约束时的极值的方法。
这种方法可以将一个有n个变量与k个约束条件的最优化问题转换为一个解有n + k个变量的方程组的解的问题。
这种方法中引入了一个或一组新的未知数，即拉格朗日乘数，又称拉格朗日乘子，或拉氏乘子，它们是在转换后的方程，
即约束方程中作为梯度（gradient）的线性组合中各个向量的系数。
https://zh.wikipedia.org/wiki/%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E4%B9%98%E6%95%B0
"""
import copy

import math


class MaxEntropy:

    def __init__(self, max_iter=1000, wi_bound=0.005):
        self._samples = []
        self.ys = set()                     # 标签集合，相当去去重后的y
        self.fi_expects = []

        self.xy_id_dict = dict()
        self.id_xy_dict = dict()
        self.xy_count_dict = dict()         # key为(x,y)，value为出现次数

        self.max_count = 0                 # 最大特征数

        self.ws = []
        self.last_ws = []

        self.max_iter = max_iter
        self.wi_bound = wi_bound                      # 收敛条件

    def load_data(self, dataset):
        self._samples = copy.deepcopy(dataset)
        for item in self._samples:
            X, y = item[1:], item[0]
            self.ys.add(y)
            # (x, y) 对应的数量 v(X=x, Y=y)
            for x in X:
                if (x, y) in self.xy_count_dict:
                    self.xy_count_dict[(x,y)] += 1
                else:
                    self.xy_count_dict[(x, y)] = 1

            # why?
            self.max_count = max([len(sample)-1 for sample in self._samples])

            # 初始化参数矩阵
            self.ws = [0] * self.get_xy_N()
            self.last_ws = self.ws

            self.fi_expects = [0] * self.get_xy_N()
            for i, xy in enumerate(self.xy_count_dict):
                # 每一种xy的组合占总的N的比例
                self.fi_expects[i] = self.xy_count_dict[xy] / self.get_N()
                self.xy_id_dict[xy], self.id_xy_dict[i] = i, xy

    def _convergence(self):
        """ 判断是否收敛 """
        for last_wi, wi in zip(self.last_ws, self.ws):
            if abs(last_wi - wi) > self.wi_bound:
                return False
        return True

    def fit(self):
        """ 拟合，训练ws """
        iter_time = 0
        for iter_time in range(self.max_iter):
            self.last_ws = self.ws[:]
            for i in range(self.get_xy_N()):
                # 计算第i个特征的模型期望
                ep = self._model_expect(i)
                # 更新参数, ep 向 fi_expects的方向更新
                self.ws[i] += math.log(self.fi_expects[i]/ep)/self.max_count
            if self._convergence():
                return iter_time
        return iter_time

    def predict(self, X):
        zx_value = self._z_x_function(X)
        result = {}
        for y in self.ys:
            ss = 0
            for x in X:
                if (x,y) in self.xy_id_dict:
                    id = self.xy_id_dict[(x,y)]
                    ss += self.ws[id]
            p_y_x = math.exp(ss) / zx_value
            result[y] = p_y_x
        return result

    def _model_expect(self, i):
        """
        计算第i个特征的模型期望
        :param i:
        :return:
        """
        x, y = self.id_xy_dict[i]
        ep = 0
        for sample in self._samples:
            if x in sample:
                p_y_x = self._p_y_x_function(y, sample)
                ep += p_y_x
        return ep / self.get_N()

    def _p_y_x_function(self, y, X):
        """
        P_w(y / x)， 根据公式计算
        :param y: 取得y值
        :param X: 在条件X的情况下
        :return:
        """
        zx_value = self._z_x_function(X)
        ss = 0
        for x in X:
            if (x, y) in self.xy_id_dict:
                i = self.xy_id_dict[(x, y)]
                ss += self.ws[i]
        pyx_value = math.exp(ss) / zx_value
        return pyx_value

    def _z_x_function(self, X):
        """
        模拟Z(x) 函数， y在所有可能的ys中
        :param X:
        :return:
        """
        zx = 0
        for y in self.ys:
            ss = 0
            for x in X:
                # fi(x,y), x,y满足在这个字典里，然后这里根据公式，得是同一个i下标，在这里，不是这个i下标的，fi都为0
                if (x,y) in self.xy_id_dict:
                    i = self.xy_id_dict[(x,y)]
                    ss += self.ws[i]
            zx += math.exp(ss)
        return zx

    def get_N(self):
        return len(self._samples)

    def get_xy_N(self):
        return len(self.xy_count_dict)


ds = [['no', 'sunny', 'hot', 'high', 'FALSE'],
           ['no', 'sunny', 'hot', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'high', 'FALSE'],
           ['yes', 'rainy', 'mild', 'high', 'FALSE'],
           ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
           ['no', 'rainy', 'cool', 'normal', 'TRUE'],
           ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
           ['no', 'sunny', 'mild', 'high', 'FALSE'],
           ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
           ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
           ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
           ['yes', 'overcast', 'mild', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
           ['no', 'rainy', 'mild', 'high', 'TRUE']]
if __name__ == '__main__':
    model = MaxEntropy()
    model.load_data(ds)
    model.fit()
    x = ['overcast', 'mild', 'high', 'FALSE']
    print(model.predict(x))