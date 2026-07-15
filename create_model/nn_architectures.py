"""Keras model builders and NN input/optimizer helpers.

Split out of the original create_model.py - function bodies are verbatim
(only module-flag references now go through create_model.state).
"""
import csv
import ctypes
from datetime import timedelta, datetime
import gc
import json
import logging
import os
import pickle
import shutil
from dateutil import parser
import subprocess
import joblib
import pycaret
from pycaret.regression import *
from pycaret.internal.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pytz
import traceback
import psutil
from pathlib import Path

import tensorflow
from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, SimpleRNN, LSTM, GRU, Bidirectional, Dropout, Input, Reshape
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, get as get_optimizer
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.backend import floatx
import tensorflow.keras.models as keras_models
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from keras_tuner import Hyperband, HyperParameters
import time

from utils import NetworkUtils, TimeUtils
from tune_grids import PYCARET_REGRESSION_TUNE_GRIDS
from subprocess_manager import run_tuning_and_ensemble_nn_with_subprocess, run_tuning_and_ensemble_with_subprocess, run_tuning_with_subprocess, run_ensemble_with_subprocess

from . import state
from .constants import *


# Create NN
def create_nn_model(hp, shape):
    model = Sequential()

    activation = hp.Choice('activation', ['relu', 'tanh'])

    # First layer (always present)
    units_hidden1 = hp.Int('units_hidden1', min_value=32, max_value=256, step=32)
    model.add(Dense(units_hidden1, activation=activation, input_shape=shape))

    # Optional second layer
    use_second_layer = hp.Boolean('use_second_layer')
    if use_second_layer:
        units_hidden2 = hp.Int('units_hidden2', min_value=16, max_value=128, step=16)
        model.add(Dense(units_hidden2, activation=activation))

    # Optional third layer
    use_third_layer = hp.Boolean('use_third_layer')
    if use_third_layer:
        units_hidden3 = hp.Int('units_hidden3', min_value=8, max_value=64, step=8)
        model.add(Dense(units_hidden3, activation=activation))

    # Output layer
    model.add(Dense(1))

    # Tune the optimizer and learning rate
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    # Tune the batch size
    hp.Choice("batch_size", [16, 32, 64, 128])

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae'])

    # Add a custom attribute to identify the model
    model.model_name = "nn_model"
    model.hp = hp
    model.shape = shape

    return model


# Create CNN
def create_cnn_model(hp, shape):
    model = Sequential()
    num_conv_layers = hp.Int('num_conv_layers', 1, 3)

    for i in range(num_conv_layers):
        filters = hp.Int(f'filters_{i}', min_value=32, max_value=256, step=32)
        max_kernel_size = min(5, shape[0])
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[k for k in [1,2,3,5] if k <= max_kernel_size])
        
        if i == 0:
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu', input_shape=shape))
        else:
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu'))
        
        if shape[0] > 1:
            model.add(MaxPooling1D(pool_size=2))
        
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(dropout_rate))
    
    model.add(Flatten())
    
    dense_units = hp.Int('dense_units', min_value=64, max_value=512, step=64)
    model.add(Dense(dense_units, activation='relu'))
    
    model.add(Dense(1))
    
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    optimizer = Adam(learning_rate) if optimizer_choice == 'adam' else RMSprop(learning_rate)
    
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    model.model_name = "cnn_model"
    model.hp = hp
    model.shape = shape

    return model


# Create RNN
def create_rnn_model(hp, shape):
    model = Sequential()

    # Tune number of RNN layers
    num_layers = hp.Int('num_rnn_layers', 1, 3)

    for i in range(num_layers):
        units = hp.Int(f'units_rnn_{i}', min_value=32, max_value=256, step=32)
        return_sequences = (i < num_layers - 1)  # Only return sequences for intermediate layers

        if i == 0:
            model.add(SimpleRNN(units=units, activation='relu', input_shape=shape, return_sequences=return_sequences))
        else:
            model.add(SimpleRNN(units=units, activation='relu', return_sequences=return_sequences))

        # Optional dropout
        if hp.Boolean(f'use_dropout_{i}'):
            dropout_rate = hp.Float(f'dropout_rate_{i}', 0.1, 0.5, step=0.1)
            model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))

    # Tune the optimizer and learning rate
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    # Tune the batch size
    hp.Choice("batch_size", [16, 32, 64, 128])

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae'])

    # Add a custom attribute to identify the model
    model.model_name = "rnn_model"
    model.hp = hp
    model.shape = shape

    return model


# Create GRU
def create_gru_model(hp, shape):
    model = Sequential()

    # Choose number of GRU layers
    num_layers = hp.Int('num_gru_layers', 1, 3)

    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        units = hp.Int(f'units_gru_{i}', min_value=32, max_value=256, step=32)
        dropout_rate = hp.Float(f'dropout_rate_{i}', 0.0, 0.5, step=0.1)

        if i == 0:
            model.add(GRU(units=units, activation='relu', input_shape=shape, return_sequences=return_sequences))
        else:
            model.add(GRU(units=units, activation='relu', return_sequences=return_sequences))

        model.add(Dropout(dropout_rate))  # Add dropout after each GRU layer

    # Output layer
    model.add(Dense(1))

    # Tune the optimizer and learning rate
    optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate)

    # Tune the batch size
    hp.Choice("batch_size", [16, 32, 64, 128])

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae'])

    # Add a custom attribute to identify the model
    model.model_name = "gru_model"
    model.hp = hp
    model.shape = shape

    return model


# Create bi-LSTM
def create_lstm_model(hp, shape):
    model = Sequential()
    model.add(Input(shape=shape))

    # Tune number of LSTM layers
    num_layers = hp.Int('num_lstm_layers', 1, 3)

    for i in range(num_layers):
        units = hp.Int(f'units_lstm_{i}', min_value=32, max_value=256, step=32)
        use_bidirectional = hp.Boolean(f'bidir_layer_{i}')

        return_sequences = (i < num_layers - 1)  # Return sequences for all but last layer

        lstm_layer = LSTM(units=units, activation='relu', return_sequences=return_sequences)
        layer = Bidirectional(lstm_layer) if use_bidirectional else lstm_layer

        # if i == 0:
        #     model.add(layer if not isinstance(layer, Bidirectional) else layer)
        #     model.layers[-1]._batch_input_shape = (None,) + shape  # Set input shape explicitly
        # else:
        #     model.add(layer)

        model.add(layer)

        # Optional dropout after each layer
        if hp.Boolean(f'use_dropout_{i}'):
            dropout_rate = hp.Float(f'dropout_rate_{i}', 0.1, 0.5, step=0.1)
            model.add(Dropout(dropout_rate))

    #Output layer
    model.add(Dense(1))

    # Tune the optimizer and learning rate
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    # Tune the batch size
    hp.Choice("batch_size", [16, 32, 64, 128])

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae'])
    
    # Add a custom attribute to identify the model
    model.model_name = "lstm_model"
    model.hp = hp
    model.shape = shape

    return model



# keras Models available functions to train => add new models here!!!
Model_functions = {
    "nn_model" : create_nn_model,
    "cnn_model" : create_cnn_model,
    "rnn_model" : create_rnn_model,
    "gru_model" : create_gru_model,
    "lstm_model" : create_lstm_model
}


def prepare_lstm_data(data):
    # Prepare LSTM input explicitly
    reshaped_data = data.reshape(
        data.shape[0], 1, data.shape[1]
    )

    return reshaped_data


# Builds model for keras tuner
def model_builder_with_shape(model_func, shape):
    def build_model(hp):
        tensorflow.keras.backend.clear_session()
        return model_func(hp, shape)
    return build_model


# Get safe model name for saving
def safe_model_name(obj, fallback="model"):
    if hasattr(obj, "model_name") and obj.model_name:
        return str(obj.model_name)

    if hasattr(obj, "name") and obj.name:
        return str(obj.name)

    # Keras model class name fallback
    return obj.__class__.__name__.lower()


# helper in case hp is dict or keras tuner object
def hp_get(hp, key, default):
    if hp is None:
        return default
    if isinstance(hp, dict):
        return hp.get(key, default)
    return hp.values.get(key, default)


# helper to rebuild optimizer from config, if model was saved and loaded
def build_optimizer(hp):
    if not isinstance(hp, (dict, type(None))) and not hasattr(hp, "values"):
        raise TypeError(f"Unsupported hp type: {type(hp)}")

    opt = hp_get(hp, "optimizer", "adam")
    lr  = hp_get(hp, "learning_rate", 1e-3)

    if opt == "adam":
        return Adam(learning_rate=lr)
    elif opt == "rmsprop":
        return RMSprop(learning_rate=lr)
    elif opt == "sgd":
        return SGD(learning_rate=lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt}")


def fresh_optimizer_from_model_or_hp(base_model, hp):
    try:
        return get_optimizer(base_model.optimizer.get_config())
    except Exception:
        return build_optimizer(hp)


# Ensures X has the correct shape for the given model: LSTM: (N, 1, features), Others: (N, features)    
def adapt_X_for_model(model, X):
    X = np.asarray(X)

    if not hasattr(model, "input_shape") or model.input_shape is None:
        return X

    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) == 2:
        return X

    if len(input_shape) == 3:
        _, d1, d2 = input_shape

        if d1 == 1 and X.ndim == 2:
            return X.reshape(X.shape[0], 1, X.shape[1])

        if d2 == 1 and X.ndim == 2:
            return X.reshape(X.shape[0], X.shape[1], 1)

        return X

    return X
