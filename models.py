from __future__ import absolute_import, division, print_function, unicode_literals, generators

import os
import pathlib
import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.layers import Dense, LSTM, LeakyReLU, Conv1D, MaxPooling1D, Dropout
from keras.models import Sequential


def regression_model1():
	regression_model = Sequential()
    regression_model.add(LSTM(100, activation='linear', input_shape=(None, 1)))
    regression_model.add(LeakyReLU())
    regression_model.add(Dense(50, activation='linear'))
    regression_model.add(LeakyReLU())
    regression_model.add(Dense(25, activation='linear'))
    regression_model.add(LeakyReLU())
    regression_model.add(Dense(12, activation='linear'))
    regression_model.add(LeakyReLU())
    regression_model.add(Dense(units=1, activation='linear'))
    regression_model.add(LeakyReLU())


    regression_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    regression_model.summary()
    return regression_model

 def regression_model2():
 	 regression_model2 = Sequential()
    regression_model2.add(Conv1D(32, 5, activation='linear', input_shape=(None, 1)))
    regression_model2.add(LeakyReLU())
    regression_model2.add(MaxPooling1D(3))
    regression_model2.add(Conv1D(32, 5, activation='linear'))
    regression_model2.add(LeakyReLU())
    regression_model2.add(LSTM(500, activation='linear'))
    regression_model2.add(LeakyReLU())
    regression_model2.add(Dense(250, activation='linear'))
    regression_model2.add(LeakyReLU())
    regression_model2.add(Dense(25, activation='linear'))
    regression_model2.add(LeakyReLU())
    regression_model2.add(Dense(12, activation='linear'))
    regression_model2.add(LeakyReLU())
    regression_model2.add(Dense(units=1, activation='linear'))
    regression_model2.add(LeakyReLU())


    regression_model2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    regression_model2.summary()
    return regression_model2

def regression_model3():
	regression_model3 = Sequential()
    regression_model3.add(Conv1D(32, 5, activation='linear', input_shape=(None, 1)))
    regression_model3.add(LeakyReLU())
    regression_model3.add(MaxPooling1D(3))
    regression_model3.add(Conv1D(32, 5, activation='linear'))
    regression_model3.add(LeakyReLU())
    regression_model3.add(LSTM(500, activation='linear', return_sequences=True))
    regression_model3.add(LeakyReLU())
    regression_model3.add(LSTM(250, activation='linear'))
    regression_model3.add(LeakyReLU())
    regression_model3.add(Dense(250, activation='linear'))
    regression_model3.add(LeakyReLU())
    regression_model3.add(Dense(25, activation='linear'))
    regression_model3.add(LeakyReLU())
    regression_model3.add(Dense(12, activation='linear'))
    regression_model3.add(LeakyReLU())
    regression_model3.add(Dense(units=1, activation='linear'))
    regression_model3.add(LeakyReLU())


    regression_model3.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    regression_model3.summary()
    return regression_model3
