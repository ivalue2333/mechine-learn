# 衡量标准

[TOC]



## Synopsis

## Refers

ml的谷歌速成学堂：https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall?hl=zh-cn

## Details

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

