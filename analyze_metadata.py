#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential

train_data = pd.read_csv('metadata.tsv', sep='\t', header=None)

X = train_data.iloc[:, 1:]
Y = train_data.iloc[:, 0]

print(X.shape)
print(Y.shape)

model = Sequential()
model.add(Dense(7, input_dim=7, kernel_initializer='normal',
                activation='relu'))
model.add(Dense(3, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=100, batch_size=32)

print('RMSE', math.sqrt(model.evaluate(X, Y)))

X_test = np.concatenate((
    np.random.randint(0, 1, 100).reshape(-1, 1),
    np.random.randint(1, 4, 100).reshape(-1, 1),
    np.random.randint(0, 3, 100).reshape(-1, 1),
    np.random.randint(1, 201, 100).reshape(-1, 1),
    np.random.randint(1, 101, 100).reshape(-1, 1),
    np.random.randint(1, 11, 100).reshape(-1, 1),
    np.random.randint(1, 11, 100).reshape(-1, 1),
), axis=1)

Y_test = model.predict(X_test)

print(X_test.shape)
print(Y_test.shape)

predictions = np.concatenate((Y_test, X_test), axis=1)

sorted_predictions = predictions[predictions[:, 0].argsort()]

np.set_printoptions(precision=3, suppress=True)
print(sorted_predictions[:10])

