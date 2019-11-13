from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])

""" 1 sample, 5 time steps, and 1 feature, """
X = seq.reshape(1, length, 1)
y = seq.reshape(1, length, 1)
# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 1000
# create LSTM
model = Sequential()
""" 5*1 x 1*5, 输出 """
""" return_sequences=True表示每一个time_step都输出，就转换为3维了 1 * 5 * 5
 """
model.add(LSTM(n_neurons, input_shape=(length, 1), return_sequences=True))

""" 5 * 5 x 5 * 1 = 5 * 1 """
""" 在这里表示输出一个 1*5*1的矩阵， (length, 1) """

""" TimeDistributed 表示
    它强调，我们打算从序列中为输入中的每个时间步输出一个时间步。
    在这种情况下，我们必须一次处理输入序列的五个时间步。
    时间分布通过将相同的密集层（相同的权重）应用于LSTMS输出，一次一个时间步来实现这一技巧。
    这样，输出层只需要一个到每个LSTM单元的连接（加上一个偏差）。
 """
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
print(result.shape)
for value in result[0, :, 0]:
    print('%.1f' % value)
