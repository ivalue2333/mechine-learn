import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

""" 数据量小，不一定成功的 """

""" 1. prepare data """
X = np.linspace(-1, 1, 200)
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))

X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]
# plt.scatter(X, Y)
# plt.show()

""" 2. define model """
model = Sequential()
model.add(Dense(units=1, input_dim=1))


""" 3. choose loss function and optimizing method """
model.compile(loss="mse", optimizer="sgd")


""" 4. train """
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)


""" 5. test """
print('-' * 10)
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost', cost)
W, B = model.layers[0].get_weights()
print("W = ", W, "B = ", B)


""" 6. predict """
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()