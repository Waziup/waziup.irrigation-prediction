"""Quarantined dead code kept for reference - no live call sites (verified).

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
from .cleaning import resample
from .evaluation import eval_approach_mix, evaluate_against_testset, evaluate_against_testset_nn
from .features import split_by_ratio
from .nn_ensemble import compare_nn_ensembles
from .nn_training import init_nn_subprocess_tuning_and_ensemble, prepare_data_for_cnn2, prepare_future_values, save_models_nn, train_best_nn, train_nn_models, tune_model_nn
from .orchestration import data_pipeline
from .prediction import align_with_latest_sensor_values, calc_threshold, compare_train_predictions_cols, create_future_values, generate_predictions, generate_predictions_nn
from .pycaret_models import create_and_compare_ensemble, create_and_compare_model_reg, init_pycaret_subprocess_tuning_and_ensemble, load_models, save_models, train_best, tune_models, tune_one_model
from .soil import add_volumetric_col_to_df


# Resample and interpolate
def check_gaps(data):
    mask = data.isna().any()
    print(mask)
    if (mask.any()):
        data.dropna(inplace=True)
        data = resample(data)
        return data.interpolate(method='linear')
    else:
        return data


# Volumetric water content => Not called
def soil_tension_to_volumetric_water_content(soil_tension, soil_water_retention_curve):
    """
    Convert soil tension (kPa) to volumetric water content (fraction) using a given soil-water retention curve.
    
    Parameters:
        soil_tension (float): Soil tension value in kPa.
        soil_water_retention_curve (list of tuples): A list of tuples containing points on the soil-water retention curve.
            Each tuple contains two elements: (soil_tension_value, volumetric_water_content_value).
    
    Returns:
        float: Volumetric water content as a fraction (between 0 and 1).
    """

    # Find the two points on the curve that bound the given soil tension value
    lower_point, upper_point = None, None
    for tension, water_content in soil_water_retention_curve:
        if tension <= soil_tension:
            lower_point = (tension, water_content)
        else:
            upper_point = (tension, water_content)
            break
    
    # If the soil tension is lower than the first point on the curve, use the first point
    if lower_point is None:
        return soil_water_retention_curve[0][1]
    
    # If the soil tension is higher than the last point on the curve, use the last point
    if upper_point is None:
        return soil_water_retention_curve[-1][1]
    
    # Interpolate to find the volumetric water content at the given soil tension
    tension_diff = upper_point[0] - lower_point[0]
    water_content_diff = upper_point[1] - lower_point[1]
    interpolated_water_content = lower_point[1] + ((soil_tension - lower_point[0]) / tension_diff) * water_content_diff
    
    return interpolated_water_content


# VWC with log scale => Not called
def soil_tension_to_volumetric_water_content_log(soil_tension, soil_water_retention_curve):
    # Transform the tension and content values to logarithmic space
    tensions_log = np.log10([point[0] for point in soil_water_retention_curve])
    content_log = np.log10([point[1] for point in soil_water_retention_curve])

    # Interpolate in logarithmic space
    interpolated_content_log = np.interpolate(np.log10(soil_tension), tensions_log, content_log)

    # Transform back to linear space
    interpolated_content = 10 ** interpolated_content_log
    
    return interpolated_content


# This function will align values sensor values with weather data from API -> not used any more
def align_retention_curve_with_api(data, data_weather_api, currentPlot):
    # Check config being loaded, otherwise read it
    if not currentPlot.config:
        currentPlot.config = currentPlot.read_config() # this is just for the case of returning to index, after settings was created/changed
    soil_water_retention_tupel_list = [(float(dct['Soil tension']), float(dct['VWC'])) for dct in currentPlot.config['Soil_water_retention_curve']]
    # Sort the soil-water retention curve points by soil tension in ascending order => TODO: NOT EFFICIENT HERE, move out
    sorted_curve = sorted(soil_water_retention_tupel_list, key=lambda x: x[0])
    # compare weatherdata from past against measured values more expressive:
    mean_recorded_sensor_values = data["rolling_mean_grouped_soil_vol"].mean()
    mean_open_meteo_past_vol = data_weather_api["Soil_moisture_0-7"].mean()
    factor = mean_open_meteo_past_vol / mean_recorded_sensor_values
    print("mean_recorded_sensor_values: ", mean_recorded_sensor_values, " mean_open_meteo_past_vol: ", mean_open_meteo_past_vol, " factor: ", factor)
    # Multiply the second column by factor
    modified_curve = [(x, y * factor) for x, y in sorted_curve]

    return modified_curve


# Calculate time since pump was on (time since last irrigation)
# TODO: need to be included, but is not tested yet
def hours_since_pump_was_turned_on(df):    
    # Find the index of rows where pump state is 1
    pump_on_indices = df[df['pump_state'] == 1].index

    # Initialize a new column with NaN values
    df['rows_since_last_pump_on'] = float('nan')

    # Iterate over pump_on_indices and update the new column
    for i in range(len(pump_on_indices)):
        if i == 0:
            # If it's the first occurrence, update with the total rows
            df.loc[:pump_on_indices[i], 'rows_since_last_pump_on'] = len(df)
        else:
            # Update with the difference in rows since the last occurrence
            df.loc[pump_on_indices[i - 1] + 1:pump_on_indices[i], 'rows_since_last_pump_on'] = \
                (pump_on_indices[i] - pump_on_indices[i - 1] - pd.Timedelta(seconds=1)) / pd.Timedelta('1 hour')

    # Fill NaN values with 0 for rows where pump state is 1
    df['rows_since_last_pump_on'] = df['rows_since_last_pump_on'].fillna(0).astype(int)

    return df


# Normalize the data in min - max approach from 0 - 1
def normalize(data):
    # feature scaling
    data.describe()
    
    # Min-Max Normalization
    df = data.drop(['Time','rolling_mean_grouped_soil', 'hour', 'minute', 'date', 'month'], axis=1)
    df_norm = (df-df.min())/(df.max()-df.min())
    df_norm = pd.concat([df_norm, data['Time'],data['hour'], data['minute'], data['date'], data['month'], data.rolling_mean_grouped_soil], 1)

    # bring back to order -> not important -> will not work in production
    data = data[['Time', 'hour', 'minute', 'date', 'month', 'grouped_soil', 
                 'grouped_resistance', 'grouped_soil_temp', 'rolling_mean_grouped_soil', 
                 'rolling_mean_grouped_soil_temp', 
                 '638de53168f31919048f189d, 63932c6e68f319085df375b5; B2_solar_x2_03, Soil_tension', 
                 '638de53168f31919048f189d, 63932c6e68f319085df375b7; B2_solar_x2_03, Resistance', 
                 '638de53168f31919048f189d, 63932c6e68f319085df375b9; B2_solar_x2_03, Soil_temperature', 
                 '638de56a68f31919048f189e, 63932ce468f319085df375cb; B3_solar_x1_02, Resistance', 
                 '638de57568f31919048f189f, 63932c3668f319085df375ab; B4_solar_x1_03, Soil_tension', 
                 '638de57568f31919048f189f, 63932c3668f319085df375ad; B4_solar_x1_03, Resistance', 
                 '638de57568f31919048f189f, 63932c3668f319085df375af; B4_solar_x1_03, Soil_temperature'
                 ]]

    return df_norm


# Split dataset into train and test set by date
def split_data_by_date(data, split_date):
    return data[data['Time'] <= split_date].copy(), \
           data[data['Time'] >  split_date].copy()


# Delete the ones that are non consecutive
def delete_nonconsecutive_rows(df, column_name, min_consecutive):
    arr = df[column_name].to_numpy()
    i = 0
    while i < len(arr) - 1:
        if arr[i+1] == arr[i] + 1:
            start_index = i
            while i < len(arr) - 1 and arr[i+1] == arr[i] + 1:
                i += 1
            end_index = i
            if end_index - start_index + 1 < min_consecutive:
                df = df.drop(range(start_index, end_index+1))
        i += 1
    return df


# Create visual representation of irrigation times
def highlight(data_plot, ax, neg_slope):
    for index, row in neg_slope.iterrows():
        current_index = int(row['index'])
        #print(current_index)
        ax.axvspan(current_index-10, current_index+10, facecolor='pink', edgecolor='none', alpha=.5)


# Create ranges to remove from data
def create_split_tuples(df, indices_to_omit):
    # Sort the indices in ascending order
    indices_to_omit = sorted(indices_to_omit)

    # Create a list of index ranges to remove
    ranges_to_remove = []
    start_idx = None
    for idx in indices_to_omit:
        if start_idx is None:
            start_idx = idx
        elif idx == start_idx + 1:
            start_idx = idx
        else:
            ranges_to_remove.append((int(start_idx), int(idx-1)))
            start_idx = idx
    if start_idx is not None:
        ranges_to_remove.append((int(start_idx), df.index.max()))
        
    print("Irrigation times to be omitted: ", ranges_to_remove)
    print("type: ", type(ranges_to_remove[0][0]))

    return ranges_to_remove


# Split data to split dataframes
def split_dataframe(df, index_ranges):
    dfs = []
    for i, (start, end) in enumerate(index_ranges):
        if index_ranges[i][1]-index_ranges[i][0] < 50:
            continue
        else:
            dfs.append(df.iloc[index_ranges[i][0]:index_ranges[i][1]])
            
    return dfs


# Main function to split dataframes -> Obsolete
def split_sub_dfs(data, data_plot):
    # calculate slope of "rolling_mean_grouped_soil"
    f = data.rolling_mean_grouped_soil
    data['gradient'] = np.gradient(f)
    
    # create dataframe with downward slope
    neg_slope = pd.DataFrame({"index":[],
                             "rolling_mean_grouped_soil":[],
                             "gradient":[]}
                            )
    
    for index, row in data.iterrows():
        if row['gradient'] < -0.07: 
            #print(index, row['rolling_mean_grouped_soil'], row['gradient'])
            current_series = pd.Series([int(round(index,0)), row['rolling_mean_grouped_soil'],
                                        row['gradient']], index=['index', 
                                                                 'rolling_mean_grouped_soil', 
                                                                 'gradient']).to_frame().T
            neg_slope = neg_slope.append(current_series)
    
    # dont ask, I love pandas^^
    neg_slope_2 = pd.DataFrame({'index':[], 'rolling_mean_grouped_soil':[], 'gradient': []})
    neg_slope_2 = pd.concat([neg_slope_2, neg_slope], ignore_index=True)
    neg_slope = neg_slope_2
    
    # Delete the ones that are non consecutive
    neg_slope = delete_nonconsecutive_rows(neg_slope, 'index', 5)
    with open('output.txt', 'w') as f:
        print(neg_slope, file=f)
    
    # visualize areas with downward slope
    ax = data_plot.drop(['Time'], axis=1).plot()
    highlight(data_plot, ax, neg_slope)
    ax.figure.suptitle("""Irrigation times highlighted\n\n""", fontweight ="bold") 
    ax.figure.savefig('irrigation_times_temp.png', dpi=400)
    
    # convert to numpy array and to int
    neg_slope_indices = neg_slope['index'].to_numpy()
    neg_slope_indices = neg_slope_indices.astype(np.int32)
    
    # Create ranges to remove from data
    tuples_to_remove = create_split_tuples(data, neg_slope_indices)
    
    # Split data to split dataframes
    sub_dfs = split_dataframe(data, tuples_to_remove) 
    
    # print dataframes
    with open('output.txt', 'a') as f:
        print("There are ", len(sub_dfs), " dataframes now.", file=f)
        for sub_df in sub_dfs:
            print(sub_df.head(1), file=f)
            print(len(sub_df), file=f)
            sub_df.drop(['Time', 'hour', 'minute', 'date', 'month'], axis=1).plot()
            
    return data, sub_dfs


# Find global max and min in all sub_dfs and cut them from min to max 
# => train data will start with min and end with max 
def format_begin_end(sub_dfs):
    cut_sub_dfs = []
    for i in range(len(sub_dfs)):
        # reset "new" index
        sub_dfs[i] = sub_dfs[i].reset_index()
        
        # index
        global_min_index = sub_dfs[i]['rolling_mean_grouped_soil'].idxmin()
        global_max_index = sub_dfs[i]['rolling_mean_grouped_soil'].idxmax()
        # value
        global_min = sub_dfs[i]['rolling_mean_grouped_soil'].min()
        global_max = sub_dfs[i]['rolling_mean_grouped_soil'].max()
        
        print(i,": ",global_min_index, "value:", global_min, global_max_index, "value:", global_max, "length:", global_max_index-global_min_index)
        print(i,": ",global_min_index, "value:", sub_dfs[i]['rolling_mean_grouped_soil'][global_min_index], global_max_index, "value:", sub_dfs[i]['rolling_mean_grouped_soil'][global_max_index], "length:", global_max_index-global_min_index)
        
        
        cut_sub_dfs.append(sub_dfs[i].iloc[global_min_index:global_max_index])
    
    # Print them    
    for df in cut_sub_dfs:
        df.drop(['index','Time','hour', 'minute', 'date', 'month'],axis=1).plot()
        
    # Preserve old index and clean
    for i in range(len(cut_sub_dfs)):
        cut_sub_dfs[i] = cut_sub_dfs[i].reset_index()
        # clean dataframe
        cut_sub_dfs[i] = cut_sub_dfs[i].drop(['level_0'], axis=1)
        cut_sub_dfs[i] = cut_sub_dfs[i].rename(columns={'index':'orig_index'})
        
    # Print head of dfs
    i = 1
    with open('output.txt', 'a') as f:
        for df in cut_sub_dfs:
            print("Dataframe: ", i, file=f)
            i+=1
            print(df.iloc[:1], file=f)
        
    return cut_sub_dfs


# Combine them to one dataframe
def combine_dfs(cut_sub_dfs):
    # save all dataframes to one and rename
    df_comb = pd.DataFrame()
    for i in range(len(cut_sub_dfs)):
        # copy elements to one df_comb
        
        df_comb = pd.concat([df_comb, cut_sub_dfs[i]['orig_index']], axis=1)
        df_comb = pd.concat([df_comb, cut_sub_dfs[i]['rolling_mean_grouped_soil']], axis=1)
        
        df_comb = df_comb.rename(columns={'orig_index':'orig_index_' + str(i)})
        df_comb = df_comb.rename(columns={'rolling_mean_grouped_soil':'rolling_mean_grouped_soil_' + str(i)})
    
    # series are not of same length => visualized here!
    df_comb.drop(['orig_index_0', 'orig_index_1', 'orig_index_2', 'orig_index_3'],axis=1).plot()
    
    return df_comb


# Create model in pycaret (time series) -> currently that approach is not used, TODO: investigate again?
def create_and_compare_model_ts(cut_sub_dfs):    
    # call setup of pycaret
    exp=[]
    for i in range(len(cut_sub_dfs)):
        exp.append(TSForecastingExperiment())
        
        # check the type of exp
        type(exp[i])
        
        # init setup on exp
        exp[i].setup(
            cut_sub_dfs[i], 
            target = 'rolling_mean_grouped_soil', 
            enforce_exogenous = False, 
            fold_strategy='timeseries',
            fh = fh, 
            session_id = 123, 
            fold = 3,
            ignore_features = ['Time', 'orig_index', 'gradient']
            #numeric_imputation_exogenous = 'mean'
        )
    
    with open('output.txt', 'a') as f:
        # check statistical tests on original data
        for i in range(len(cut_sub_dfs)):
            print("This is the", i, "part of the data:", file=f)
            print(exp[i].check_stats(), file=f)
            
    best = []
    for i in range(len(cut_sub_dfs)):
        print("This is for the", i, "part of the dataset: ")
        best.append(exp[i].compare_models(
            n_select = 5, 
            fold = 3, 
            sort = 'R2',
            verbose = state.Verbose_logging, 
            exclude=['lar_cds_dt','auto_arima','arima'],
            #include=['lr_cds_dt', 'br_cds_dt', 'ridge_cds_dt', 
            #        'huber_cds_dt', 'knn_cds_dt', 'catboost_cds_dt']
        ))
    
    with open('output.txt', 'a') as f:       
        for i in range(len(best)):
            print("\n The best model, for cut_sub_dfs[", i,"] is:", file=f)
            print(best[i][0], file=f)
    
    return exp, best


# prepare data for (conv) neural nets and other model architechtures -> OBSOLETE: use prepare_data_for_cnn2 instead
def prepare_data_for_cnn(data, target_variable):
    # to rangeindex => do not use timestamps!
    data = data.reset_index(drop=False, inplace=False)
    #data.rename(columns={'index': 'Timestamp'}, inplace=True)

    # Drop non important
    data_nn = data.drop(columns=To_be_dropped, axis=1, inplace=False) #dropping yields worse results (val_loss in training)

    # Split the dataset into features (X) and target variable (y)
    X = data_nn.drop(target_variable, axis=1)  # Assuming 'target_variable' is the target variable
    y = data_nn[target_variable]

    # Determine the split point (80% training, 20% testing)
    split_point = int(len(X) * 0.8)

    # Split the data into training and testing sets based on the split point
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ensure input data is correctly reshaped for Conv1D
    X_train_cnn = X_train_scaled[..., np.newaxis]
    X_test_cnn = X_test_scaled[..., np.newaxis]

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_train_cnn, X_test_cnn, scaler


def default_hp_nn(): # TODO: wire those into training function
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

    return hp


# Symmetric mean absolute percentage error (SMAPE or sMAPE) is an accuracy measure based on percentage (or relative) errors.
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


# Quantile Loss can be defined as a custom loss function, which can be trained to minimize Quantile Loss. If the set percentile values are close to 0 or 1, the training results do not follow the trend of the training data and are relatively flat
def quantile_loss(y_true, y_pred, q):
    e = y_true - y_pred
    return np.mean(np.maximum(q * e, (q - 1) * e))


# Calculate mse, rmse, mae, mpe, r2, quantile_loss, SMAPE
def evaluate_target_variable_nd(series1, series2, quantiles=[0.2, 0.4, 0.6, 0.8]):
    # Step 1: Compute the differences between the two series
    differences = series1 - series2
    
    # Step 2: Compute MSE, RMSE, MPE, and R2 score based on the differences
    mse = mean_squared_error(series1, series2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(series1, series2)
    mpe = np.mean(differences / series1) * 100
    r2 = r2_score(series1, series2)
    smape_score = smape(series1, series2)
    quantile_losses = [quantile_loss(series1, series2, q) for q in quantiles]
    # Compute the average quantile loss
    avg_quantile_loss = np.mean(quantile_losses)
    
    # Print the results
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Percentage Error (MPE): {mpe:.2f}%")
    print(f"R-squared (R2) Score: {r2:.2f}")
    print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape_score:.2f}%")
    for q, loss in zip(quantiles, quantile_losses):
        print(f"Quantile Loss (q={q}): {loss:.2f}")
    print(f"Average Quantile Loss: {avg_quantile_loss:.2f}")

    return mse, rmse, mae, mpe, r2, smape_score, quantile_losses, avg_quantile_loss


# TODO:wire this
def compare_models_on_test(nn_models, ZZ, Z_cnn):
    best_r2 = -1000
    best_model_index  = -1
    for i in range(len(nn_models)):
        print("This is the " + str(i+1) + ". model in the pipeline")
        # print summary
        if isinstance(nn_models[i], Sequential) or isinstance(nn_models[i], Model):
            nn_models[i].summary()
        else:
            print('For this model there is no summary\n')
            
        try:
            # Make predictions
            predictions = nn_models[i].predict(Z_cnn)
        
            # Evaluate model performance
            print("Metrics:", predictions.shape)
            evaluation_result = evaluate_target_variable_nd(ZZ.values, predictions.reshape(predictions.shape[0], predictions.shape[1]))
            
            # Check if evaluation result is not None
            if evaluation_result is not None:
                mse, rmse, mae, mpe, r2, smape_score, quantile_losses, avg_quantile_loss = evaluation_result
                # select best
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_index = i
            else:
                print("Evaluation result is None. Skipping metrics printing.")

            print("\n*****************************************************************")
        except Exception as e:
            print(f"Is not available for the model. {e}")
            
    if best_model_index != -1:
        print("The best model after evaluation on unseen data during training is:", best_model_index)
        best_model = nn_models[best_model_index]
    else:
        print("No model met the criteria for selection.")   

    return best_model_index


# Obsolete: use eval_approach_mix instead
def eval_approach(results, results_nn, metrics = 'mae'):
    use_pycaret = True
    index = -1
    if metrics == 'mae':
        best_result = float('inf')
        for i in range(len(results)):
            current = results[i]['results'][0]
            if current < best_result:
                index = i
                best_result = current
                use_pycaret = True
        for i in range(len(results_nn)):
            current = results_nn[i]['results'][0]
            if current < best_result:
                index = i
                best_result = current
                use_pycaret = False
    else:
        print('Unknown metircs specified in eval_approach.')

    return index, use_pycaret


def analyze_performance_old(exp, best):
    # plot forecast
    for i in range(len(best)):
        print("This is for model: ",i)
        exp.plot_model(best[i], plot = 'forecast', save = True)
        #.save('Plot_in_testset_'+str(i)+'.png', format='png')
        
        print("After testset: For the dataset:",i)
        exp[i].plot_model(best[i], plot = 'forecast', data_kwargs = {'fh' : 500}, save = True)


# Helper to get R2 from evaluate_model->pycaret eval model is not stable across all models
def get_r2(exp, model):
    if model is None:
        return -9999  # automatically worst
    df = exp.evaluate_model(model)
    if df is None:
        print(f"Warning: evaluate_model returned None for {model}")
        return -9999
    return df.loc["R2", "Mean"] if "R2" in df.index else -9999


# Mighty main fuction ;) -> TODO: create more meaningful logs
def main_old(plot) -> int:
# (flag now lives in state module)
    
    # Prevents multiple training or prediction at the same time
    while state.Currently_active:
        print(f"[{plot.user_given_name}] Waiting for resources to be released. Another training or prediction is already running.")
        time.sleep(Resource_wait_time_seconds)
        
    # Before training starts, lock the resource    
    state.Currently_active = True
        
    # Check version of pycaret, should be >= 3.0
    print("Check version of pycaret:", pycaret.__version__, "should be >= 3.0")
    # Load config, to get latest changes before training starts
    plot.config = plot.read_config()
    
    if state.SkipDataPreprocessing:
        # Load data from disk
        plot.data = pd.read_csv('data/debug/debug_data.csv').set_index('Timestamp')
        plot.data.index = pd.to_datetime(plot.data.index, utc=True).tz_convert(TimeUtils.Timezone)
        train, val, test = split_by_ratio(plot.data)
        #X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_train_cnn, X_test_cnn, scaler = prepare_data_for_cnn(plot.data, 'rolling_mean_grouped_soil')
        train, val, test, X_train, X_val, X_test, y_train, y_val, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_train_cnn, X_val_cnn, X_test_cnn, scaler = prepare_data_for_cnn2(plot, train, val, test, 'rolling_mean_grouped_soil') #testing consistant val dataset for comparison
    else:
        # Data preparation pipeline: get config, fetch, align, clean, sample....
        train, val, test, X_train, X_val, X_test, y_train, y_val, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_train_cnn, X_val_cnn, X_test_cnn, scaler = data_pipeline(plot)

    # Debug mode -> skips training and uses debug.csv
    if state.SkipTraining:
        debug_df = pd.read_csv('data/debug/debug_predictions.csv').set_index('Timestamp')
        debug_df.index = pd.to_datetime(debug_df.index, utc=True).tz_convert(TimeUtils.Timezone)

        return 12, pd.Timestamp(datetime.now().replace(microsecond=0, second=0, minute=0)), debug_df
    
    # Start training pipeline: setup, train models the best ones to best-array
    # Classical regression
    #exp, best = create_and_compare_model_ts(cut_sub_dfs)
    exp, best = create_and_compare_model_reg(train)
    # NN
    nn_models = train_nn_models(X_train, X_val, y_train, y_val, X_train_scaled, X_val_scaled, X_train_cnn, X_val_cnn, plot.user_given_name)
    
    # Save the best models for further evaluation -> could be also omitted, since best_models are more important TODO
    # Classical regression:
    model_names = save_models(plot.user_given_name, exp, best, f'models/{plot.user_given_name}/intermediate_models/pycaret/soil_tension_prediction_')
    # NN: (print eval(on X_test) and save to disk)
    save_models_nn(plot.user_given_name, nn_models, f'models/{plot.user_given_name}/intermediate_models/nn/soil_tension_prediction_', None)
    
    # Load regression model from disk, if there was a magical error => TODO: nn not implemented yet, but also useless, omit, because it would stop before, surround more with try except
    try:
        best
    except NameError:
        best = load_models(model_names)

    # Evaluate with testset
    # Classical regression
    best_eval, results = evaluate_against_testset(plot, test, exp, best)
    # NN
    # Force ensemble or single usage for DEBUG purposes
    #plot.ensemble = False
    best_eval_nn, results_nn = evaluate_against_testset_nn(plot, nn_models, X_test_scaled, y_test)

    # Decide which approach is best according to evaluation metric of one metric (old approach)
    # index, plot.use_pycaret = eval_approach(results, results_nn, 'r2')
    
    # Decide which approach is best according to evaluation metric mix of mae, mpe, r2
    index, plot.use_pycaret = eval_approach_mix(results, results_nn, weights=None) # TODO: check again, define weights?

    # TODO FROM HERE ON THIS COULD BE ALSO CAPSULATED IN SEPARATE FUNCTION
    # Force pycaret or nn usage for DEBUG purposes
    #plot.use_pycaret = False

    # Train best model on whole dataset (without skipping "test-set")
    if plot.use_pycaret:
        # Classical regression
        plot.best_model, plot.best_exp = train_best(best_eval, plot.data)
    else:
        # NN
        best_model_nn, X_train_scaled, X_val_scaled, y_train, y_val = train_best_nn(best_eval_nn, plot.data, plot.user_given_name)

    # TODO: Save "best" models here?
    
    # Create future value set to feed new data to model
    future_features = create_future_values(plot.data, plot)
    # Compare dataframes cols to be sure that they match, otherwise drop
    future_features = compare_train_predictions_cols(train, future_features)
    # NN
    if not plot.use_pycaret:
        Z, Z_scaled, Z_cnn = prepare_future_values(scaler, future_features, X_train.columns)


    if plot.use_pycaret:
        # Tune hyperparameters of one or the 3 best models
        try:
            gc.collect()
            # Store original best model before tuning, to fallback in case of error
            best_model_before_tuning = plot.best_model
        
            if state.Use_subprocess:
                # Perform entire tuning pipeline in subprocess
                plot.best_model = init_pycaret_subprocess_tuning_and_ensemble(plot.user_given_name, plot.best_exp, plot.best_model, plot.ensemble)
            else:
                # Perform tuning in the main process
                if plot.ensemble:
                    plot.best_model = tune_models(plot.best_exp, plot.best_model)
                else:
                    plot.best_model = tune_one_model(plot.best_exp, plot.best_model)
                
                # Save best tuned pycaret model
                model_names = save_models(plot.user_given_name, plot.best_exp, plot.best_model, f'models/{plot.user_given_name}/tuned_models/pycaret/best_soil_tension_prediction_')
        except Exception as e:
            print(f"[{plot.user_given_name}] Error during tuning: {e}, using the original model.")
            plot.best_model = best_model_before_tuning  # Keep original model if tuning fails

        
        # After tuning
        #best_model_after_tuning = plot.best_exp.compare_models()
        
        # Manually compare metrics
        #metrics_before_tuning = plot.best_exp.get_metrics(model=best_model_before_tuning)
        #metrics_after_tuning = plot.best_exp.get_metrics(model=best_model_after_tuning)

        # Compare relevant metrics
        # compare_df = pd.DataFrame({
        #     'Metric': metrics_before_tuning['Metric'],
        #     'Before_Tuning': metrics_before_tuning['Value'],
        #     'After_Tuning': metrics_after_tuning['Value']
        # })

        # print(compare_df)
        
        # Ensemble, Stacking & Blending
        if plot.ensemble and not state.Use_subprocess:  # If ensemble is enabled and not using subprocess, create ensemble in main process
            gc.collect()
            plot.best_model = create_and_compare_ensemble(plot.user_given_name, plot.best_exp, plot.best_model)
            # Save best pycaret ensemble model
            model_names = save_models(plot.user_given_name, plot.best_exp, plot.best_model, f'models/{plot.user_given_name}/ensemble_models/pycaret/best_soil_tension_prediction_')


        # Create predictions to forecast values
        future_features_without_index = future_features.reset_index(drop=True, inplace=False)
        future_features_without_index = future_features_without_index.rename(columns={'index': 'Timestamp'}, inplace=False)
        plot.predictions = generate_predictions(plot.best_model, plot.best_exp, future_features_without_index)
        plot.predictions['Timestamp'] = future_features.index  # Copy index to column
        plot.predictions = plot.predictions.set_index("Timestamp")  # Set Timestamp as index
    # Neural Networks
    else:
        # Tune and create ensemble model
        if state.Use_subprocess and plot.ensemble:
            # Perform entire tuning and ensemble pipeline in subprocess
            plot.best_model = init_nn_subprocess_tuning_and_ensemble(plot.user_given_name, X_train_scaled, y_train, X_val_scaled, y_val, best_model_nn)
            # Save best ensemble model
            #save_models_nn(plot.user_given_name, plot.best_model, f'models/{plot.user_given_name}/ensemble_models/nn/best_tuned_soil_tension_ensemble_prediction_nn_model_', None)
        elif plot.ensemble and not state.Use_subprocess:
            tuned_best_models = []
            tuned_best_hps = []

            # Tune top3 models first
            for i in range(len(best_model_nn)):
                tuned_best, best_hp = tune_model_nn(X_train_scaled, y_train, X_val_scaled, y_val, best_model_nn[i])
                tuned_best_models.append(tuned_best)
                tuned_best_hps.append(best_hp)
            plot.best_model = tuned_best_models

            # Save tuned best models
            save_models_nn(plot.user_given_name, plot.best_model, f'models/{plot.user_given_name}/tuned_models/nn/best_soil_tension_prediction_', tuned_best_hps)

            # Create and compare ensemble techniques
            results_ensemble = compare_nn_ensembles(plot.best_model, tuned_best_hps, X_train_scaled, y_train, X_val_scaled, y_val, metric="r2", bagging_rounds=5, stacking_folds=5, verbose=state.Verbose_logging)
            plot.best_model = results_ensemble["best_predictor"]
            print(f"[{plot.user_given_name}] Selected NN ensemble method: {results_ensemble['best_name']} with scores: {results_ensemble['scores']}") #already printed prior

            # Save best ensemble model
            save_models_nn(plot.user_given_name, plot.best_model, f'models/{plot.user_given_name}/ensemble_models/nn/best_tuned_soil_tension_ensemble_prediction_nn_model_', tuned_best_hps)
        else:
            # Only tune best model
            plot.best_model, _ = tune_model_nn(X_train_scaled, y_train, X_val_scaled, y_val, best_model_nn)
            # Save tuned best model
            save_models_nn(plot.user_given_name, plot.best_model, f'models/{plot.user_given_name}/tuned_models/nn/best_soil_tension_prediction_', None)

        #DEBUG: load best model from disk->only for debug reasons to avoid tuning training TODO: implement as a proper fallback
        #plot.best_model, tuned_best_hps = load_models_nn(f'models/{plot.user_given_name}/best_models/nn/')

        # Generate predictions with NN model
        plot.predictions = generate_predictions_nn(plot.best_model, Z_scaled, future_features.index[0], future_features.index[-1])


    # Ensure the index of plot.predictions is datetime with the same timezone
    if plot.predictions.index.tz is None:
        #plot.predictions.index = pd.to_datetime(plot.predictions.index).tz_localize('UTC').tz_convert(TimeUtils.Timezone)
        plot.predictions.index = pd.to_datetime(plot.predictions.index).tz_localize(TimeUtils.Timezone)
    else:
        plot.predictions.index = plot.predictions.index.tz_convert(TimeUtils.Timezone)

    # Create a Timestamp from the current date and time (without microseconds, seconds, and minutes)
    #current_time = pd.Timestamp(datetime.now().replace(microsecond=0, second=0, minute=0))
    current_time = pd.Timestamp.now(tz=TimeUtils.Timezone).floor('H')

    # Now, slice the predictions DataFrame based on the timestamp
    # Cut passed time from predictions
    if not plot.load_data_from_csv:
        plot.predictions = plot.predictions.loc[current_time:]
        #plot.predictions = plot.predictions.loc[pd.Timestamp((datetime.now()).replace(microsecond=0, second=0, minute=0)).tz_localize(TimeUtils.Timezone):]    
    
    # Align predictions with historical data -> TODO: dodgy fix, only trigger in case of bad performance? DEBUG
    align_with_latest_sensor_values(plot)
    plot.predictions['smoothed_values'] = plot.predictions['prediction_label']

    # Calculate when threshold will be meet
    plot.threshold_timestamp = calc_threshold(plot.predictions, 'smoothed_values', plot)

    # Add volumetric water content
    if plot.sensor_kind == 'tension':
        plot.predictions = add_volumetric_col_to_df(plot.predictions, "smoothed_values", plot)

    # After finished job set active to false
    state.Currently_active = False

    # Return last accumulated reading and threshold timestamp
    return plot.data['rolling_mean_grouped_soil'][-1], plot.threshold_timestamp, plot.predictions
