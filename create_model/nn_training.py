"""NN training, Hyperband tuning, persistence and subprocess orchestration.

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
from .nn_architectures import Model_functions, adapt_X_for_model, create_cnn_model, create_gru_model, create_lstm_model, create_nn_model, create_rnn_model, model_builder_with_shape, prepare_lstm_data, safe_model_name
from .nn_ensemble import EnsemblePredictor
from .runtime import HardCleanupCallback, MemoryLimitCallback, MemoryLimitReachedError, TimeLimitCallback, free_memory


def prepare_data_for_cnn2(
    plot,
    train_df,
    val_df,
    test_df,
    target_variable,
    training=True
):
    """
    Prepare already-split train/val/test datasets for NN models
    WITHOUT introducing data leakage.
    """

    # ---------------------------------------
    # RESET INDEX
    # ---------------------------------------
    train_df = train_df.reset_index(drop=False, inplace=False)
    val_df   = val_df.reset_index(drop=False, inplace=False)
    test_df  = test_df.reset_index(drop=False, inplace=False)

    # ---------------------------------------
    # DROP UNUSED COLUMNS
    # ---------------------------------------
    train_df = train_df.drop(columns=To_be_dropped, axis=1, errors='ignore')
    val_df   = val_df.drop(columns=To_be_dropped, axis=1, errors='ignore')
    test_df  = test_df.drop(columns=To_be_dropped, axis=1, errors='ignore')

    # ---------------------------------------
    # SPLIT FEATURES / TARGET
    # ---------------------------------------
    X_train = train_df.drop(columns=[target_variable])
    y_train = train_df[target_variable]

    X_val = val_df.drop(columns=[target_variable])
    y_val = val_df[target_variable]

    X_test = test_df.drop(columns=[target_variable])
    y_test = test_df[target_variable]

    # ---------------------------------------
    # SCALING
    # FIT ONLY ON TRAIN, AND ONLY WHEN TRAINING
    # ---------------------------------------
    if training:
        plot.data_scaler = StandardScaler()
        X_train_scaled = plot.data_scaler.fit_transform(X_train)
    else:
        # Prediction path: the frozen model learned under this scaler's statistics, so the
        # stored scaler must be reused as-is. Refitting here (the old behavior - the fit ran
        # unconditionally) fed the model inputs on a drifting scale as new data accumulated
        # between retrainings.
        if getattr(plot, 'data_scaler', None) is None or not hasattr(plot.data_scaler, 'mean_'):
            raise ValueError("No fitted scaler found in plot object - run a training first.")
        X_train_scaled = plot.data_scaler.transform(X_train)

    # IMPORTANT:
    # NEVER FIT AGAIN
    X_val_scaled = plot.data_scaler.transform(X_val)
    X_test_scaled = plot.data_scaler.transform(X_test)

    # ---------------------------------------
    # CNN SHAPES
    # ---------------------------------------
    X_train_cnn = X_train_scaled[..., np.newaxis]
    X_val_cnn = X_val_scaled[..., np.newaxis]
    X_test_cnn = X_test_scaled[..., np.newaxis]

    return (
        X_train,
        X_val,
        X_test,

        y_train,
        y_val,
        y_test,

        X_train_scaled,
        X_val_scaled,
        X_test_scaled,

        X_train_cnn,
        X_val_cnn,
        X_test_cnn,

        plot.data_scaler
    )


# Models being trained on different architectures
def train_nn_models(X_train, X_val, y_train, y_val, X_train_scaled, X_val_scaled, X_train_cnn, X_val_cnn, plot_name):
    
    # Create an array to store all the models
    nn_models = []

    # Create neural network

    # Create a dummy HyperParameters object with fixed values
    hp = HyperParameters()
    hp.Fixed('activation', 'relu')
    hp.Fixed('units_hidden1', 128)
    hp.Fixed('use_second_layer', True)
    hp.Fixed('units_hidden2', 64)
    hp.Fixed('use_third_layer', True)
    hp.Fixed('units_hidden3', 32)
    hp.Fixed('optimizer', 'adam')
    hp.Fixed('learning_rate', 0.001)


    # Call the model function with the hp object and the input shape
    input_shape = (X_train.shape[1],)
    model_nn = create_nn_model(hp, shape=input_shape)

    # Train the model
    print('Will now train a Neural net (NN), with the following hyperparameters: ' + str(hp.values))
    history_nn = model_nn.fit(
        X_train_scaled, 
        y_train, 
        epochs=50, 
        batch_size=32, 
        validation_data=(X_val_scaled, y_val)
    )
    # Append for comparison
    nn_models.append(model_nn)

    # Create conv neural network

    # Create a dummy HyperParameters object with fixed values
    hp = HyperParameters()
    hp.Fixed('num_conv_layers', 2)

    # First conv layer
    hp.Fixed('filters_0', 64)
    hp.Fixed('kernel_size_0', 3)

    # Second conv layer
    hp.Fixed('filters_1', 64)
    hp.Fixed('kernel_size_1', 3)

    # Dropout rate (shared across all layers in your model)
    hp.Fixed('dropout_rate', 0.3)

    # Dense layer
    hp.Fixed('dense_units', 128)

    # Optimizer and learning rate
    hp.Fixed('optimizer', 'adam')
    hp.Fixed('learning_rate', 0.001)

    # Call the model function with the hp object and the input shape
    input_shape = (1, X_train_cnn.shape[1])
    model_cnn = create_cnn_model(hp, shape=input_shape)
    
    # Train the model
    print('Will now train a Convolutional neural net (CNN), with the following hyperparameters: ' + str(hp.values))
    history_cnn = model_cnn.fit(X_train_cnn[:, np.newaxis, :], 
                                y_train, 
                                epochs=50, 
                                batch_size=32, 
                                validation_data=(X_val_scaled[:, np.newaxis, :], y_val)
    )

    # Append for comparison
    nn_models.append(model_cnn)

    # Create RNN model

    # Create a dummy HyperParameters object with fixed values
    hp = HyperParameters()
    # Number of RNN layers
    hp.Fixed('num_rnn_layers', 2)

    # Layer 0
    hp.Fixed('units_rnn_0', 64)
    hp.Fixed('use_dropout_0', True)
    hp.Fixed('dropout_rate_0', 0.3)

    # Layer 1
    hp.Fixed('units_rnn_1', 32)
    hp.Fixed('use_dropout_1', False)  # No dropout in the second layer

    # Optimizer settings
    hp.Fixed('optimizer', 'adam')
    hp.Fixed('learning_rate', 0.001)

    input_shape = (1, X_train.shape[1])
    model_rnn = create_rnn_model(hp, shape=input_shape)

    # Train the model
    print('Will now train a Recurrent neural network (RNN), with the following hyperparameters: ' + str(hp.values))
    history_rnn = model_rnn.fit(X_train_scaled[:, np.newaxis, :], 
                                y_train, 
                                epochs=50, 
                                batch_size=32, 
                                validation_data=(X_val_scaled[:, np.newaxis, :], y_val)
    )

    # Append for comparison
    nn_models.append(model_rnn)

    # Create GRU model

    # Create a dummy HyperParameters object with fixed values
    hp = HyperParameters()
    hp.Fixed('num_gru_layers', 2)
    hp.Fixed('units_gru_0', 64)
    hp.Fixed('dropout_rate_0', 0.2)
    hp.Fixed('units_gru_1', 32)
    hp.Fixed('dropout_rate_1', 0.2)
    hp.Fixed('optimizer', 'adam')
    hp.Fixed('learning_rate', 1e-3)

    input_shape = (1, X_train.shape[1])
    model_gru = create_gru_model(hp, shape=input_shape)
    # Train the model
    print('Will now train a Gated Recurrent Unit neural network (GRU), with the following hyperparameters: ' + str(hp.values))
    history_gru = model_gru.fit(X_train_scaled[:, np.newaxis, :], 
                                y_train, 
                                epochs=50, 
                                batch_size=32, 
                                validation_data=(X_val_scaled[:, np.newaxis, :], y_val)
    )

    # Append for comparison
    nn_models.append(model_gru)

    # LSTM architecture

    # Create a dummy HyperParameters object with fixed values
    hp = HyperParameters()
    # Architecture
    hp.Fixed('num_lstm_layers', 2)
    hp.Fixed('units_lstm_0', 64)
    hp.Fixed('bidir_layer_0', True)
    hp.Fixed('use_dropout_0', True)
    hp.Fixed('dropout_rate_0', 0.2)

    hp.Fixed('units_lstm_1', 32)
    hp.Fixed('bidir_layer_1', False)
    hp.Fixed('use_dropout_1', True)
    hp.Fixed('dropout_rate_1', 0.2)

    # Optimizer and learning rate
    hp.Fixed('optimizer', 'adam')
    hp.Fixed('learning_rate', 0.001)

    # Prepare model shape
    input_shape = (1, X_train.shape[1])

    # Create model architecture
    model_lstm = create_lstm_model(hp, shape=input_shape)

    # Prepare data for LSTM input explicitly
    X_train_lstm = prepare_lstm_data(X_train_scaled)
    X_val_lstm = prepare_lstm_data(X_val_scaled)

    # Train the model
    print('Will now train a Long short-term memory neural network (LSTM), with the following hyperparameters: ' + str(hp.values))
    history_bilstm = model_lstm.fit(X_train_lstm, 
                                    y_train, 
                                    epochs=50, 
                                    batch_size=32, 
                                    validation_data=(X_val_lstm, y_val)
    )
    
    # Append for comparison
    nn_models.append(model_lstm)
    
    # Save training history plots and jsons
    if state.Verbose_logging:
        for history, name in zip(
            [history_nn, history_cnn, history_rnn, history_gru, history_bilstm],
            ['nn_model', 'cnn_model', 'rnn_model', 'gru_model', 'lstm_model']
        ):
            plot_history_png(history, filename=f'models/{plot_name}/intermediate_models/nn/soil_tension_prediction_{name}_{datetime.now()}.png')
            save_history_json(history, filename=f'models/{plot_name}/intermediate_models/nn/soil_tension_prediction_{name}_{datetime.now()}.json')

    return nn_models


# Save NN models
def save_models_nn(plot_name, nn_models, path_to_save, nn_hps = None):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

    model_paths = []  # To keep track of saved model paths for reference

    # type check for array -> convert
    if not isinstance(nn_models, list):
        nn_models = [nn_models]

    # Save multiple trained models and HPS for future use
    for i in range(len(nn_models)):
        # Get a safe model name
        model_name = safe_model_name(nn_models[i])
        base_name = f"{path_to_save}{i}_{model_name}_{plot_name}" 

        # Check if the model is a Keras model
        if isinstance(nn_models[i], tensorflow.keras.Model):
            try:
                # Save the trained models for future use
                p = base_name + ".keras"
                nn_models[i].save(p)
                model_paths.append(p)
                print(f"[OK] Saved Keras model: {p}")
            except Exception as e:
                print(f"[FAIL] Could not save Keras model {p}: {e}")

            if nn_hps is not None:
                try:
                    if nn_hps is not None and nn_hps[i] is not None:
                        with open(base_name + "_hps.json", "w") as f:
                            json.dump(nn_hps[i].values, f, indent=2)
                except Exception as e:
                    print(f"[Fail] HP save failed ({model_name}): {e}")

        # Ensemble models
        elif isinstance(nn_models[i], EnsemblePredictor):
            try:
                ensemble_meta = {
                    "model_name": nn_models[i].model_name,
                    "type": "ensemble",
                    "note": "This is a meta-predictor. Base models are saved separately."
                }

                with open(base_name + "_ensemble_meta.json", "w") as f:
                    json.dump(ensemble_meta, f, indent=2)

                # Save the ensemble predictor itself
                p = base_name + "_ensemble.joblib"
                joblib.dump(nn_models[i], p)
                model_paths.append(p)

                print(f"[OK] Saved ensemble metadata: {p}")

            except Exception as e:
                print(f"[FAIL] Could not save ensemble predictor: {e}")  
        # Unknown model type
        else:
            print(
                f"[SKIP] Object of type {type(nn_models[i])} cannot be saved "
                f"(model_name={getattr(nn_models[i], 'model_name', 'unknown')})"
            ) 

    return model_paths


# Load NN models
def load_models_nn(input_path):
    models = []
    best_hps = []

    # should be capsulated in functions for loading single models and loading ensembles, but for now we keep it simple and just check if path is dir or file
    if os.path.isdir(input_path):
        # Sort to ensure model i matches hyperparameter i
        for file in sorted(os.listdir(input_path)):
            # Keras models
            if file.endswith(".h5") or file.endswith(".keras"):
                path = os.path.join(input_path, file)
                model = keras_models.load_model(path)

                # Parse name: index_modelname_plot.h5
                parts = file.replace(".h5", "").replace(".keras", "").split("_", 2)
                if len(parts) >= 2:
                    model.model_name = parts[1]

                models.append(model)
                print(f"Loaded NN model: {file}")

                # Attempt to load corresponding hyperparameters JSON
                base_name, _ = os.path.splitext(file)
                hp_file = f"{base_name}_hps.json"
                hp_path = os.path.join(input_path, hp_file)
                if os.path.exists(hp_path):
                    try:
                        with open(hp_path, "r") as f:
                            hps = json.load(f)
                        best_hps.append(hps)
                    except Exception as e:
                        print(f"Failed to load hyperparameters for {file}: {e}")
                        best_hps.append(None)
                else:
                    best_hps.append(None)
            # Ensemble models
            elif file.endswith(".joblib"):
                path = os.path.join(input_path, file)
                try:
                    ensemble_model = joblib.load(path)
                    models.append(ensemble_model)
                    print(f"Loaded ensemble model: {file}")
                except Exception as e:
                    print(f"Failed to load ensemble model {file}: {e}")
    else:
        # Keras model
        if input_path.endswith(".h5") or input_path.endswith(".keras"):
                model = keras_models.load_model(input_path)

                # Parse name: index_modelname_plot.h5
                parts = input_path.replace(".h5", "").replace(".keras", "").split("_", 2)
                if len(parts) >= 2:
                    model.model_name = parts[1]

                models.append(model)
                print(f"Loaded NN model: {input_path}")
                # Hyperparameters are not needed when loading single models
        # Ensemble model
        elif input_path.endswith(".joblib"):
            try:
                ensemble_model = joblib.load(input_path)
                models.append(ensemble_model)
                print(f"Loaded ensemble model from: {input_path}")
            except Exception as e:
                print(f"Failed to load ensemble model from: {input_path}: {e}")

    return models, best_hps


# Plot training history of neural nets
def plot_history_png(history, filename, y_max=2.0):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.figure(figsize=(10, 5))

    losses = history.history['loss'] + history.history.get('val_loss', [])
    y_upper = min(np.percentile(losses, 95), y_max)

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')

    if 'mae' in history.history:
        plt.plot(history.history['mae'], label='mae')
        plt.plot(history.history['val_mae'], label='val_mae')

    plt.xlabel("Epoch")
    plt.ylabel("Loss / MAE")
    plt.ylim(0, y_upper)
    plt.legend()
    plt.grid()

    plt.savefig(filename, dpi=200)
    plt.close()


# Save training history as JSON
def save_history_json(history, filename):
    import json
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Convert values to float and prepare JSON structure
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    # Save file
    with open(filename, "w") as f:
        json.dump(hist, f, indent=4)


# Train the best model on the full dataset TODO: metrics bad check again!
def train_best_nn(best_eval, data, plot_name):
    # Reset index to range index
    data = data.reset_index(drop=False, inplace=False)
    data = data.rename(columns={'index': 'Timestamp'}, inplace=False)

    # Drop non-important columns
    data_nn = data.drop(To_be_dropped, axis=1, inplace=False)  # Drop irrelevant columns

    # Split the dataset into features (X) and target variable (y)
    X = data_nn.drop('rolling_mean_grouped_soil', axis=1)  # Assuming 'soil_tension' is the target variable
    y = data_nn['rolling_mean_grouped_soil']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features using training data only -> avoid data leakage, created new scaler
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_val_scaled = final_scaler.transform(X_val)

    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    if not isinstance(best_eval, list):
        function_name = best_eval.model_name
        if function_name in Model_functions:   
            # Create the best model
            model = Model_functions[function_name](best_eval.hp, best_eval.shape)
            print(f"Will train the '{model.model_name}' as best model for neural nets.")

            # Adapt input data
            X_t = adapt_X_for_model(model, X_train_scaled)
            X_v = adapt_X_for_model(model, X_val_scaled)

            # Train the model
            history_nn = model.fit(
                X_t,
                y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_v, y_val)#,
                #callbacks=[early_stopping],
            )
            if state.Verbose_logging:
                plot_history_png(history_nn, f'models/{plot_name}/best_models/nn/training_history_{model.model_name}_{datetime.now()}.png')
                save_history_json(history_nn, f'models/{plot_name}/best_models/nn/training_history_{model.model_name}_{datetime.now()}.json')
        else:
            print(f"Function '{function_name}' not found. Using fallback.")
            model = best_eval
    else:
        # Ensemble of multiple models
        model = []
        for i, m in enumerate(best_eval, start=1):
            function_name = m.model_name
            if function_name in Model_functions:
                # Create each model in the ensemble
                temp_model = Model_functions[function_name](m.hp, m.shape)
                print(f"Will train the '{temp_model.model_name}' as part of the {i}/{len(best_eval)} ensemble for neural nets.")

                # Adapt input data
                X_t = adapt_X_for_model(temp_model, X_train_scaled)
                X_v = adapt_X_for_model(temp_model, X_val_scaled)

                # Train the model
                history_nn = temp_model.fit(
                    X_t,
                    y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_v, y_val)#,
                    #callbacks=[early_stopping], # TODO: Evaluate early stopping, stops to early?
                )
                model.append(temp_model)
                if state.Verbose_logging:
                    plot_history_png(history_nn, f'models/{plot_name}/best_models/nn/training_history_ensemble_{i}of{len(best_eval)}_{model[i-1].model_name}_{datetime.now()}.png')
                    save_history_json(history_nn, f'models/{plot_name}/best_models/nn/training_history_ensemble_{i}of{len(best_eval)}_{model[i-1].model_name}_{datetime.now()}.json')
            else:
                print(f"Function '{function_name}' not found. Skipping this model.")
                model.append(best_eval)

    return model, X_train_scaled, X_val_scaled, y_train, y_val


# Create testset, not seen during training=>should be fur
def prepare_future_values(scaler, new_data, X_train_c):
    # Filter the list to include only columns that are present in the DataFrame
    columns_to_drop = [col for col in To_be_dropped if col in new_data.columns]
    # drop not needed features
    new_data_nn = new_data.drop(columns_to_drop, axis=1, inplace=False)
    # Align columns of df1 to match df2 -> in most cases not needed
    new_data_aligned = new_data_nn.reindex(columns=X_train_c)

    # scale testset
    Z = new_data_aligned

    Z_scaled = scaler.transform(Z)

    # numerical_columns = Z.select_dtypes(include=np.number).columns
    # Z_numerical = Z[numerical_columns]

    # # Scale the test features using the SAME scaler used for training data
    # Z_scaled = scaler.transform(Z_numerical)

    Z_cnn = Z_scaled[..., np.newaxis]

    return Z, Z_scaled, Z_cnn


# tune a neural network model with keras tuner
def tune_model_nn(X_train_scaled, y_train, X_val_scaled, y_val, best_model_nn):
    try:
        print("Tuning best NN model (", best_model_nn.model_name, ") after evaluation.")

        # # quick workaround for adjusting train shape for tuning a lstm
        # if best_model_nn.model_name in ['lstm_model']:
        #     X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        #     X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
        #     shape = (1, X_train_scaled.shape[2])  # (timesteps, features)
        # else:
        #     shape = best_model_nn.shape
        X_train_scaled = adapt_X_for_model(best_model_nn, X_train_scaled)
        X_val_scaled = adapt_X_for_model(best_model_nn, X_val_scaled)

        if state.Verbose_logging:
            print("Shapes: \n X_train shape:", X_train_scaled.shape)
            print(" Model shape:", best_model_nn.shape)

        builder = model_builder_with_shape(
            Model_functions[best_model_nn.model_name],
            best_model_nn.shape
        )

        # Initialize the Hyperband tuner
        tuner = Hyperband(
            builder,
            objective='val_mae',
            max_epochs=50,             # Tune epochs between 10 and 100 # TODO: was 100 DEBUG
            factor=3,                   # Reduces the number of epochs for each successive run, Defaults to 3, 4 would be fast, 2 is with wider scope DEBUG
            hyperband_iterations=1,     # Limits the number full hyperband runs
            directory='hyperband_dir',
            project_name='hyperband_' + best_model_nn.model_name,
            overwrite=True
        )

        # Set the max time in seconds DEBUG
        max_time_seconds = 3600 #3600 DEBUG
        time_limit_callback = TimeLimitCallback(max_time_seconds)
        # Early stopping to avoid overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        try:
            tuner.search(
                X_train_scaled,
                y_train,
                validation_data=(X_val_scaled, y_val),
                callbacks=[ # Add the time limit callback here TODO: fix: it is not working
                    time_limit_callback, 
                    early_stopping, 
                    MemoryLimitCallback(), 
                    HardCleanupCallback()
                ]  
            )
        except MemoryLimitReachedError as e:
            print(f"Tuning aborted: {e}")

        # Same clear_session+gc as before, plus malloc_trim so the Hyperband trials'
        # memory actually leaves the process (the final model is rebuilt after this)
        free_memory(clear_keras=True, label="hyperband search")

        # Print the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best hyperparameters: {best_hps.values}\nWill now rebuild and train the model with these hyperparameters...")

        # Rebuild the model using the best hyperparameters
        final_model = Model_functions[best_model_nn.model_name](best_hps, best_model_nn.shape)

        # Train the final model with the best hyperparameters on the full training data
        final_model.fit(X_train_scaled, 
                        y_train, 
                        epochs=best_hps.values.get('tuner/epochs', 50), 
                        batch_size=best_hps.values.get('batch_size', 32),
                        validation_data=(X_val_scaled, y_val)
        )

        # Compare with formerly best model -> since it is trained 
        #final_model = evaluate_against_testset_nn(best_model_nn, X_train_scaled, y_train)

        return final_model, best_hps
    
    except Exception as e:
        print(f"There was an error tuning the NN model. {e}")
        return best_model_nn


def save_weights(model, temp_dir=None):
    """
    Save Keras model weights to disk and return path.

    Args:
        model: Keras model
        temp_dir: Optional directory (Path or str)

    Returns:
        str: path to saved weights file
    """
    from pathlib import Path
    import uuid
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Ensure temp_dir exists
        if temp_dir is None:
            temp_dir = Path("./tmp_weights")
        else:
            temp_dir = Path(temp_dir) / "weights"

        temp_dir.mkdir(parents=True, exist_ok=True)

        # Unique filename (VERY important for parallel runs)
        unique_id = uuid.uuid4().hex[:8]
        model_name = getattr(model, "model_name", "model")

        weight_path = temp_dir / f"{model_name}_{unique_id}.weights.h5"

        # Ensure model is built (important for some TF models)
        if not model.built:
            try:
                model.build(model.input_shape)
            except Exception:
                logger.warning(f"Could not explicitly build model {model_name}")

        # Save weights
        model.save_weights(weight_path)

        return str(weight_path)

    except Exception as e:
        logger.error(f"Failed to save weights: {e}")
        raise


def init_nn_subprocess_tuning_and_ensemble(plot_name, X_train, y_train, X_val, y_val, models):
    """
    Orchestration wrapper to tune and create ensemble for neural networks in subprocess, 
    since it is very resource intensive and can cause memory leaks, crashes, etc. 
    It also allows to use different libraries and versions for tuning and ensemble creation, 
    without affecting the main process. 
    It saves data and models to disk, runs the tuning and ensemble creation in a separate process, 
    and loads the final model back into the main process. 
    It also handles exceptions and fallbacks gracefully. 
    """
    model_configs = []
    
    temp_dir = Path("./tmp") / f"nn_{plot_name}_{datetime.now().timestamp()}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Extract config to file
    for m in models:
        model_configs.append({
            "model_name": m.model_name,
            "shape": m.shape,
            "weights_path": save_weights(m, temp_dir),
            "hp_values": m.hp.values if hasattr(m, "hp") and m.hp is not None else None
        })
        
    # Save model configs (NOT models)
    with open(temp_dir / "model_configs.json", "w") as f:
        json.dump(model_configs, f)
    
    # Save data
    np.save(temp_dir / "X_train.npy", X_train)
    np.save(temp_dir / "y_train.npy", y_train)
    np.save(temp_dir / "X_val.npy", X_val)
    np.save(temp_dir / "y_val.npy", y_val)

    # Run tuning and ensemble creation in subprocess
    result_path = run_tuning_and_ensemble_nn_with_subprocess(temp_dir, model_configs, plot_name)

    # return the best model (first in list)
    model = load_models_nn(result_path)[0][0]

    # Delete tmp folder and all containing files
    try:
        shutil.rmtree(temp_dir)
        print(f"Deleted temp dir after successful tuning and ensemble creation: {temp_dir}")
    except Exception as e:
        print(f"Warning: Failed to delete temp dir {temp_dir}: {e}")

    return model  
