# 第一章: 统计学习方法概论

[TOC]



##  1.1统计学习

### 统计学习的特点

统计学习：也称统计机器学习。

学习：如果一个系统能够通过执行某个过程改进它的性能，

### 统计学习的对象

数据

### 统计学习的目的

数据，模型，预测

### 统计学习的方法

- 三要素
  - 模型的假设空间（学习的模型属于某个函数的集合）- 模型
  - 模型选择的准则（评价准则）- 策略
  - 模型学习的算法（最优模型选择算法）- 算法

### 统计学习的研究

统计学习方法

统计学习理论

统计学习应用

## 1.2监督学习

### 基本概念

三个空间：输入空间，输出空间，特征空间

- 三种问题
  - 回归问题：输入变量，输出变量都是连续变量的预测问题
  - 分类问题：输出变量有限个离散变量的预测问题
  - 标注问题：输入变量，输出变量都是变量序列

模型属于由输入空间到输出空间的映射的集合，这个集合就是假设空间

- 两种模型
  - 概率模型（概率分布）：P(Y/X)
  - 非概率模型（决策函数）:：Y = f(X)




## 1.3统计学习三要素

#### 假设空间

假设空间包含所有可能的条件概率或决策函数

### 策略

#### 损失函数

损失函数=代价函数(loss function = cost function)，一个意思。常用损失函数如下。

绝对损失函数：

> $$ L(Y, f(X)) = \begin{cases} 1 & Y != f(X) \\ 0 & Y = f(X) \end{cases}$$

平方损失函数：

> $ L(Y, f(X)) = (Y -f(X) )^ 2$

绝对损失函数: 

>  $ L(Y, f(X)) = |Y - f(X)|$

对数损失函数：

>  $ L(Y, f(X)) = -\log P(Y/X)$

#### 期望损失

> ​						$R_{exp}(f) = \int_{x,y} L(y, f(x) P(x,y))dxdy$

#### 经验风险最小化

经验风险，经验损失，平均损失

> ​						 $$R_{emp}(f) = \frac{1}{N} \sum_{i=1}^N L(y_i, f(x_i))$$

经验风险最小化，例如，最大似然估计。

#### 结构风险最小化

结构风险最小化是为了放置过拟合提出的策略，结构风险在经验风险上加上表示模型复杂度的正则项（regularizer）或者叫做罚项（penalty term）

> ​						$$ R_{srm}(f) = \frac{1}{N} \sum_{i=1}^N L(y_i, f(x_i)) + \lambda J(f) $$

J(f)表示模型的复杂度，和模型复杂度正相关，表示对模型复杂度的惩罚。λ>=0调节两者之间关系的系数。

结构风险最小化，例如贝叶斯估计。

#### 算法

统计学习的本质基于训练数据集，根据学习策略，从假设空间中选择最优模型。是一個最优问题，如何保证找到全局最优解，并使求解的过程非常高效，就是问题的核心。

## 1.4模型评估和模型选择

误差包括训练误差和测试误差

### 多项式函数拟合

多项式函数拟合时假设给定的数据时M次多项式函数生成的。

M阶多项式

> $f_M(x, w) = \sum_{j=0}^M w_j x^M$

平均损失

> $R_{emp}(f) = \frac{1}{2} \sum_{i=1}^N (f_M(x_i, w) - y_i) ^ 2$

系数1/2是为了方便计算。平方损失和最小二乘法是一致的。

### 过拟合

如果假设空间中存在一个真模型，所选模型的参数个数要和真模型相同，所选模型的参数向量要和真模型接近。

如果一味为了更好的拟合训练数据，导致所选模型的复杂度过高，这种现象就是过拟合。

过拟合的模型对已知数据的预测很好，对未知数据的预测很差。

避免过拟合，降低训练误差

## 1.5正则化与交叉验证

#### 正则化-regularization

#### 交叉验证-cross-validation

- 训练数据
- 验证数据
- 测试数据

## 1.6 泛化能力

泛化能力generalization ability，就是期望风险

## 1.7 生成模型与判别模型

### 生成方法

由数据学习联合概率密度P(X, Y)，然后求出条件概率分布P(Y/X)作为预测的模型

> P(Y / X) = P(X, Y) / P(X)

#### 包括

- 朴素贝叶斯
- 隐马尔科夫模型

### 判别方法

由数据直接学习决策函数f(X) 或者条件概率分布P(Y/X) 作为预测的模型。

#### 包括

- K近邻
- 感知机
- 决策树
- 逻辑斯地回归
- 最大熵
- 支持向量机
- 提升方法
- 条件随机场

## 1.8分类问题

### 准确率

> 准确率是指我们的模型预测正确的结果所占的比例

> accuracy = (number of correct predictions) / (total of predictions)

对于二元分类，也可以根据正类别和负类别按如下方式计算准确率

> accuracy = (TP + TN) / (TP + TN + FP + FN)

P表示预测为1（预测为正类别）， T 表示预测正确，TP表示预测为1与事实相符，FP表示预测为1与事实有差。

### 精确率

> 在被识别为正类别的样本中，确实为正类别的比例是多少

> precision = (TP) / (TP + FP)

### 召回率

> 在所有正类别样本中，被正确识别为正类别的比例是多少

> recall = (TP) / (TP + FN)

### F1-measure

> F1是精确率和召回率的调和平均值

> 2 / F1 = 1 / precision + 1 / recall

## 1.9标注问题

## 1.10 回归问题

回归问题的损失函数常用平方损失函数，回归问题可以使用最小二乘法解决







