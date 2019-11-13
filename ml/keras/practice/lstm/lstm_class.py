import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.models import load_model
import matplotlib.pyplot as plt


def get_data():
    df = pd.read_csv('dataset.csv')
    data = df.iloc[:, 2:10].values
    data = data[:6000]
    normalized_train_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    train_data, test_data = np.array(normalized_train_data[:5000]), np.array(normalized_train_data[5000:])
    train_x, train_y = train_data[:, :7], train_data[:, 7:]
    test_x, test_y = test_data[:, :7], test_data[:, 7:]
    return (train_x, train_y), (test_x, test_y)


class LstmDemo:
    def __init__(self, n_neurons=6, time_steps=8, input_size=7, output_size=1, n_epoch=200, n_batch=32, on_train=True):
        self.n_neurons = n_neurons
        self.time_steps = time_steps
        self.input_size = input_size
        self.output_size = output_size
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.on_train = on_train

        self._model_path = 'my_model.h5'
        self._model = self.build_model()

    def build_model(self):
        if self.on_train:
            model = Sequential()
            model.add(LSTM(units=self.n_neurons, input_shape=(self.time_steps, self.input_size), return_sequences=True))
            model.add(TimeDistributed(Dense(1)))
            model.compile(loss='mean_squared_error', optimizer='adam')
            return model

    def fit(self, train_x, train_y):

        if self.on_train:
            train_x = train_x.reshape((-1, self.time_steps, self.input_size))
            train_y = train_y.reshape((-1, self.time_steps, self.output_size))
            self._model.fit(train_x, train_y, epochs=self.n_epoch, batch_size=self.n_batch, verbose=2)
            self._model.save(self._model_path)
        else:
            self._model = load_model(self._model_path)

    def predict(self, test_x):
        test_x = test_x.reshape((-1, self.time_steps, self.input_size))
        res = self._model.predict(test_x, batch_size=self.n_batch, verbose=0)
        return res


if __name__ == '__main__':
    (tx, ty), (tex, tey) = get_data()
    ld = LstmDemo(n_epoch=200, on_train=False)
    ld.fit(tx, ty)
    y_pred = ld.predict(tex)
    y_pred = np.array(y_pred).flatten()
    tey = np.array(tey).flatten()
    # plt.plot(range(len(tey)), tey, 'r', range(len(y_pred)), y_pred, 'b--', range(len(ty)), ty, 'y')
    # plt.plot(range(len(tey)), tey, 'r', range(len(y_pred)), y_pred, 'b--')
    plt.plot(range(len(tey)), tey, 'r', label='y_target')
    plt.plot(range(len(y_pred)), y_pred, 'b--', label='y_predict')
    plt.legend(loc='upper left')
    plt.show()