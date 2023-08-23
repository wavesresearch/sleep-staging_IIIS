#!/usr/bin/env python
# coding: utf-8
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, GRU)
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

def Single_GRU(no_chs, samples, n_class=2):
    model = Sequential()
    model.add(GRU(100, input_shape=(no_chs, samples), return_sequences=True, kernel_regularizer=l2(0.0001)))
    model.add(GRU(100, return_sequences=True, kernel_regularizer=l2(0.0001)))
    model.add(GRU(100, kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(n_class, activation='softmax'))
    return model
