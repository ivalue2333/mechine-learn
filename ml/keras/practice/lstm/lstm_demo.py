import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
import matplotlib.pyplot as plt

n_neurons = 7
time_steps = 8

""" 下面这个和输入输出的维度有关 """
input_size = 7
output_size = 1

n_epoch = 200
n_batch = 32

df = pd.read_csv('dataset.csv')
data = df.iloc[:, 2:10].values
normalized_train_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
normalized_train_data = normalized_train_data[:6000]
normalized_train_data = np.array(normalized_train_data)

train_x, train_y = normalized_train_data[:, :7], normalized_train_data[:, 7:]
print(np.array(train_x).shape)
train_x = np.array(train_x).reshape((-1, time_steps, input_size))
train_y = np.array(train_y).reshape((-1, time_steps, output_size))
print(train_x.shape)
print(train_y.shape)

model = Sequential()

model.add(LSTM(units=n_neurons, input_shape=(time_steps, input_size), return_sequences=True))
model.add(TimeDistributed(Dense(1)))

model.compile(loss='mean_squared_error', optimizer='adam')


model.fit(train_x, train_y, epochs=n_epoch, batch_size=n_batch, verbose=2)

result = model.predict(train_x, batch_size=n_batch, verbose=0)

print('-' * 10)

pred = np.array(result).flatten()
print(pred.shape)
y = np.array(train_y).flatten()
print(y.shape)
print(len(pred))
plt.plot(range(len(y)), y, 'r', range(len(pred)), pred, 'b--')
plt.show()
