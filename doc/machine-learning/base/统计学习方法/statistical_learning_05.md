# 第五章：决策树

[TOC]

决策树是一种基本的分类与回归的方法。他可以被看做是if-else规则的集合，决策树学习通常包括3个步骤：特征选择，决策树生成，决策树修剪。**基于最大信息增益**

## 5.1决策树模型

分类决策树模型是一种描述对实例进行分类的树形结构，决策树由节点和有向变组成。节点有两种类型，内部节点和叶节点，内部节点表示一个特征或者树形，叶节点表示一个类。

决策树学习本质上是从训练数据中归纳出一组分类规则，与训练数据不想矛盾的分类规则可能有多个，或者一个也没有，需要找到一个和训练数据矛盾度小的分类规则，且有较好的泛化能力。

- 三个部分
  - 特征选择：单个特征的分类能力很关键
  - 决策树的生成（局部最优）
  - 决策树的剪枝（全局最优）

## 5.2特征选择

### 信息增益

熵(entropy)：随机变量不确定性的度量

$$
概率分布:p(X=x_i) = p_i  i=1, 2...n
$$

$$
熵定义： H(X) = -\sum_{i=1}^N p_i * \log p_i
$$

log以2为底，熵的单位是比特，以e为底，熵的单位是纳特(nat)

随机变量X给定的情况下，随机变量Y的条件熵。
$$
条件熵：H(Y/X) = \sum_{i=1}^N( p_i *  H (Y / X = x_i))
$$
**当熵和条件熵中的概率来自于数据估计（尤其是最大似然估计）时，所对应的熵和条件熵称为经验熵和经验条件熵 **

信息增益(information gain)：表示得知特征X而使类Y的信息的不确定性减小的程度。

特征A对训练集合D的信息增益
$$
g(D, A) = H(D) - H(D/A)
$$
**在例5.2中，通过计算各个feature的信息增益，来决定哪个特征是最优特征 **

### 信息增益比

信息增益比 
$$
gr(D, A) = g(D, A) / H(D)
$$

## 4.3决策树的生成

### ID3算法

 ID3算袪的核心是在决策树各个结点上应用信息增益准则选择特征，递归地构建决策树.具体方法是：从根结点(root node)开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征，由该特征的不同取值建立子结点；再对子结点递归地调用以上方法，构建决策树；直到所有特征的信息增益均很小或没有特征可以选择为止.最后得到一个决策树.ID3相当于用极大似然法进行概率模型的选择.

## 5.4决策树的剪枝

决策树生成只考虑了通过提高信息增益，对训练数据进行更好的拟合，而决策树的剪枝通过优化损失函数还考虑了模型复杂度（避免过拟合，过拟合就是拟合的很好，预测却往往很差）

## 5.5 CART

分类与回归树，既可以做分类也可以做回归。又称最小二乘回归树



