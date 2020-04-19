# tensorflow基础

## Synopsis

- tensorflow是一个开源的深度学习框架


- 适合所有人的开放源代码机器学习框架

## Refers

中国官网：https://tensorflow.google.cn/

tensorflow可以作什么：https://juejin.im/entry/5a69ac0b6fb9a01c9526493b

tensorflow极客学院：http://wiki.jikexueyuan.com/project/tensorflow-zh/get_started/introduction.html

## Details

### 通用步骤

- 准备数据集
  - 包括输入数据，预期输出数据Y
- 定义输入，输出占位符
  - 这样可以在"算子"中使用这些占位符，并在计算时像传入参数一般传入
- 定义隐藏层
  - 以输入占位符作为输入，定义weights的行和列数
  - 定义激励函数，激励函数用于将输出"扳弯"，如果没有激励函数，那就没有激励函数
- 定义输出层
  - 输出层以隐藏层的输出作为输入，定义输出的行和列数
  - 同样可以定义激励函数
  - 输出层的输出值即是prediction
- 计算损失loss
  - 损失就是预测值prediction和预期输出数据Y之间的误差，简单的计算如计算欧式距离等
- 定义train_step
  - train_step是这样的一个算子，他将损失值作为输入，并使用一些优化器(optimizer)，例如GradientDescentOptimizer的一些策略，如策略是最小化loss，来生成一次训练，这个训练的目的，是将输入的损失值在优化器的作用下，生成一些东西，反向传递，来优化weights和biases，使得损失是在向下降（降低）的方向移动的，这个你可以结合损失和weights的图形来得到更好的理解。
  - 所以这一步的训练就是在一次又一次地寻找最佳的weights和biases，也被称作拟合，即通过训练，使得生成的模型（你可以简单理解为一份复杂函数），和准备好的预期输出数据Y有最好的拟合（重合）
  - 然而值得注意的是，模型在拟合时存在过拟合和欠拟合的情况，所以不是所有的情况都是适合简单的最小化loss的训练方法
- 定义session变量
  - 这很重要，所有的算子都是作为session的函数变量来使用的
- 初始化变量
  - 所有的变量都需要初始化的，这很简单
- 训练
  - 使用session.run()方法，传入准备数据，执行定义好的train_step，反复训练（反复生成新的模型去拟合准备数据Y），得到模型
- 模型保存
  - 模型在session中

