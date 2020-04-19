# 神经网络

[TOC]



## Synopsis

## Refers

莫凡bilibili：https://space.bilibili.com/243821484/channel/detail?cid=26359

莫凡bilibili，有对应代码：https://www.bilibili.com/video/av16001891/?p=4

根据莫凡的视频做的笔记：https://www.jianshu.com/p/e112012a4b2d

## Details

### 神经网络在干啥

- 拟合曲线（生成模型）
- 拟合参数（对输入预测结果）

输入->神经网络黑盒（代表特征）->输出

梯度下降是关键

卷积神经网络（Convolutional Neural Networks, CNN）

循环神经网络（Recurrent Neural Network, RNN）

Autoencoder：为了压缩输入数据

GAN（Generative Adversarial Nets）：生成对抗网络

reinforcement learning：强化学习（分数为导向）

数据特征化：避免无用信息，避免重复信息，避免复杂的信息

过拟合：解决（1：增加数据量）

加速训练速度

batch normalization： 让每一层的数据在有效的范围内传递下去

L1,L2：模型过拟合，使用误差方程，将参数值计算进去，参数值太大，误差方程的值也会大