import numpy as np
# for reproducibility
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

"""
    在回归网络(regressor_example)中用到的是 model.add 一层一层添加神经层，这里的方法是直接在模型的里面加多个神经层。
    好比一个水管，一段一段的，数据是从上面一段掉到下面一段，再掉到下面一段。

    第一段就是加入 Dense 神经层。32 是输出的维度，784 是输入的维度。
    第一层传出的数据有 32 个 feature，传给激励单元，激励函数用到的是 relu 函数。
    经过激励函数之后，就变成了非线性的数据。
    然后再把这个数据传给下一个神经层，这个 Dense 我们定义它有 10 个输出的 feature。
    同样的，此处不需要再定义输入的维度，因为它接收的是上一层的输出。
    接下来再输入给下面的 softmax 函数，用来分类。
"""


# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
""" 1.1. download data """
# (60000, 28, 28)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

""" 1.2. data pre-processing """
print(X_train.shape)

# (60000, 784)
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
print('-' * 10)
print(y_test[0])
y_test = np_utils.to_categorical(y_test, num_classes=10)
print(y_test[0])


""" 2. define model """
# 多分类输出的activation 采用 softmax
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# We add metrics to get more results you want to see
""" 3. choose loss function and optimizing method """
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

""" 4. train """
model.fit(X_train, y_train, epochs=2, batch_size=32)


""" 5. test """
loss, accuracy = model.evaluate(X_test, y_test)
print('test loss: ', loss)
print('test accuracy: ', accuracy)