import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    # 升高维度 1 * 1000 -> 50 * 20
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # 50 * 20 * 1  = batch_input_shape
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


model = Sequential()
""" return_sequences: 默认 False。在输出序列中，返回单个 hidden state值还是返回全部time step 的 hidden state值。
    False 返回单个， true 返回全部。
"""
model.add(LSTM(
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    units=CELL_SIZE,            # output_dim=CELL_SIZE
    return_sequences=True,      # True: output at all time steps. False: output as last step.
    stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
))

# add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
adam = Adam(LR)
model.compile(optimizer=adam, loss='mse',)

for step in range(501):
    X_batch, Y_batch, xs = get_batch()
    cost = model.train_on_batch(X_batch, Y_batch)
    pred = model.predict(X_batch, BATCH_SIZE)

    # Y_batch: 50 * 20 * 1
    # Y_batch[0]: 20 * 1
    # pred: 50 * 20 * 1

    plt.plot(xs[0, :], Y_batch[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.1)
    if step % 10 == 0:
        print('train cost: ', cost)
