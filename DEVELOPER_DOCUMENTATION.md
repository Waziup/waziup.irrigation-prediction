# Irrigation Prediction Application - Developer Documentation

## Table of Contents
1. [Core Files](#core-files)
2. [API & Server (main.py)](#api--server-mainpy)
3. [ML Pipeline (create_model.py)](#ml-pipeline-create_modelpy)
4. [Subprocess Management (subprocess_manager.py)](#subprocess-management-subprocess_managerpy)
5. [Plot Management (plot.py, plot_manager.py)](#plot-management-plotpy-plot_managerpy)
6. [Threading (training_thread.py, prediction_thread.py)](#threading-training_threadpy-prediction_threadpy)
7. [Irrigation Control (actuation.py)](#irrigation-control-actuationpy)
8. [Utilities (utils.py)](#utilities-utilspy)
9. [HTTP Routing (usock.py)](#http-routing-usockpy)

---

## Core Files

| File | Lines | Purpose |
|------|-------|---------|
| main.py | 1074 | REST API server, route handlers, application entry point |
| create_model.py | 3350+ | ML training pipeline, neural networks, predictions |
| plot.py | 472 | Plot data model for individual fields |
| plot_manager.py | 189 | Multi-plot management |
| training_thread.py | 128 | Background training thread scheduling |
| prediction_thread.py | ~100 | Background prediction thread scheduling |
| actuation.py | 380 | Irrigation control and verification |
| utils.py | 110 | Timezone, network, and API utilities |
| usock.py | ~200 | HTTP server routing and request handling |
| subprocess_manager.py ⚠️ | 500+ | Memory-efficient subprocess execution |

---

## API & Server (main.py)

### Classes

#### `LogCleanerThread(threading.Thread)`
**Purpose**: Background thread that periodically cleans old log files  
**Location**: Lines 91-120

**Methods**:
- `__init__(file_path, age_limit_days=90, check_interval=86400, name=None)` - Initialize with log file path and cleanup parameters
- `clean_log()` - Clears log file if older than age limit
- `run()` - Thread main loop for scheduled cleanup

#### `ModelCleanerThread(threading.Thread)`
**Purpose**: Background thread that removes old model files  
**Location**: Lines 134-150

**Methods**:
- `__init__(folder_path, interval_days=7, name=None)` - Initialize with model folder path
- `run()` - Monitor and delete old models periodically

### Functions

#### `index(url, body="")`
**Purpose**: Health check endpoint  
**Returns**: (200, "Salam Goloooo", [])  
**Location**: Line 38

#### `ui(url, body='')`
**Purpose**: Serve web UI files (HTML, CSS, JS, images)  
**Parameters**:
- `url`: Request URL containing filename
- `body`: Request body (empty for GET)  
**Returns**: (status_code, file_content, [content_type])  
**Location**: Lines 47-90

#### `schedule_log_cleanup()`
**Purpose**: Start background thread for log cleanup  
**Location**: Lines 121-131

#### `delete_old_files(folder_path)`
**Purpose**: Delete files older than threshold  
**Parameters**: `folder_path` - Directory path  
**Location**: Lines 151-171

#### `schedule_model_cleanup(folder_path, interval_days=7)`
**Purpose**: Start background thread for model cleanup  
**Location**: Lines 172-179

#### `getApiUrl(url, body)`
**Purpose**: Return API endpoint base URL  
**Returns**: JSON with `api_url`  
**Location**: Line 180

#### `setPlot(url, body)`
**Purpose**: Set current active plot by tab number  
**Parameters**: `body` contains `plot_number`  
**Returns**: JSON response with status  
**Location**: Lines 200-218

#### `getPlots(url, body)`
**Purpose**: Get list of all plots with metadata  
**Returns**: JSON array of plot objects  
**Location**: Lines 219-241

#### `addPlot(url, body)`
**Purpose**: Create new plot via UI  
**Returns**: JSON with new plot number  
**Location**: Lines 242-262

#### `removePlot(url, body)`
**Purpose**: Delete plot by tab number  
**Parameters**: `body` contains `plot_number`  
**Location**: Lines 263-282

#### `setConfig(url, body)`
**Purpose**: Update plot configuration (sensors, thresholds, soil properties)  
**Parameters**: 
- `body`: JSON with all config fields
- Fields: `user_given_name`, `gps_info`, `sensors`, `irrigation`, `soil`, `training`  
**Returns**: JSON with save status  
**Location**: Lines 283-366

#### `getConfigsFromAllFiles()`
**Purpose**: Load all plot configurations from disk  
**Returns**: Dictionary of {plot_id: config_dict}  
**Location**: Lines 367-431

#### `returnConfig(url, body)`
**Purpose**: Get configuration for current plot  
**Returns**: JSON with complete plot config  
**Location**: Lines 432-512

#### `checkConfigPresent(url, body)`
**Purpose**: Check if plot has valid configuration  
**Returns**: JSON with `config_present` boolean  
**Location**: Lines 513-532

#### `checkActiveIrrigation(url, body)`
**Purpose**: Check if irrigation is currently active  
**Returns**: JSON with `is_irrigating` boolean  
**Location**: Lines 533-555

#### `extract_and_format(data, key, datatype)`
**Purpose**: Extract and format sensor data from API response  
**Parameters**:
- `data`: Raw sensor data list
- `key`: Field to extract ("temperature", "data", etc.)
- `datatype`: Type to convert to (int, float)  
**Returns**: List of formatted values  
**Location**: Lines 556-566

#### `group_sensor_data(sensor_lists, agg_func=lambda vals: sum(vals)/len(vals), resample_interval="30T")`
**Purpose**: Combine multiple sensors and aggregate/resample  
**Parameters**:
- `sensor_lists`: List of data lists from different sensors
- `agg_func`: Aggregation function (default: mean)
- `resample_interval`: Pandas resampling interval  
**Returns**: Aggregated DataFrame  
**Location**: Lines 567-616

#### `interpolate_list_with_limit(data_list, max_gap=10)`
**Purpose**: Fill missing values with linear interpolation  
**Parameters**:
- `data_list`: List of values with NaN gaps
- `max_gap`: Max consecutive NaN to interpolate  
**Returns**: List with interpolated values  
**Location**: Lines 617-645

#### `smooth_outliers(data_list, window=2, threshold=0.5)`
**Purpose**: Detect and smooth outliers using rolling window  
**Parameters**:
- `data_list`: Input list
- `window`: Rolling window size
- `threshold`: Deviation threshold  
**Returns**: Smoothed list  
**Location**: Lines 646-668

#### `extract_and_format_csv(data, key)`
**Purpose**: Extract sensor data and return as CSV format  
**Returns**: CSV string  
**Location**: Lines 669-705

#### `irrigateManually(url, body)`
**Purpose**: Manually trigger irrigation for a plot  
**Parameters**: `body` contains `amount` (liters)  
**Location**: Lines 706-723

#### `getValuesForDashboard(url, body)`
**Purpose**: Get current sensor values and predictions for UI dashboard  
**Returns**: JSON with current readings, thresholds, predictions  
**Location**: Lines 724-775

#### `getHistoricalChartData(url, body)`
**Purpose**: Get historical sensor data for chart visualization  
**Returns**: JSON with timestamps and values  
**Location**: Lines 776-839

#### `getDatasetChartData(url, body)`
**Purpose**: Get training dataset features for debugging  
**Returns**: JSON with feature names and values  
**Location**: Lines 840-883

#### `getPredictionChartData(url, body)`
**Purpose**: Get model predictions for next 7+ days  
**Returns**: JSON with timestamps and predicted soil moisture  
**Location**: Lines 884-972

#### `getThreshold(url, body)`
**Purpose**: Get current irrigation threshold and trigger time  
**Returns**: JSON with threshold value and timestamp  
**Location**: Lines 973-995

#### `getSensorKind(url, body)`
**Purpose**: Get sensor type (tension or volumetric)  
**Returns**: JSON with `sensor_kind`  
**Location**: Lines 996-1003

#### `isTrainingReady(url, body)`
**Purpose**: Check if plot has enough data and is configured  
**Returns**: JSON with `ready` boolean  
**Location**: Lines 1004-1011

#### `startTraining(url, body)`
**Purpose**: Trigger immediate model training  
**Returns**: JSON with status  
**Location**: Lines 1012-1019

#### `getCurrentPlot(url, body)`
**Purpose**: Get metadata of current active plot  
**Returns**: JSON with plot info  
**Location**: Lines 1020-1074

---

## ML Pipeline (create_model.py)

### Classes

#### `TimeLimitCallback(Callback)`
**Purpose**: Keras callback to stop training after time limit  
**Location**: Lines 125-139

**Methods**:
- `__init__(max_time_seconds=1800)` - Set time limit
- `on_epoch_end(epoch, logs=None)` - Check if time exceeded, stop if needed

#### `MemoryLimitCallback(tensorflow.keras.callbacks.Callback)`
**Purpose**: Keras callback to monitor memory during training  
**Location**: Lines 140-146

#### `EnsemblePredictor`
**Purpose**: Wrapper class for ensemble models (bagging, stacking, blending)  
**Location**: Lines 2750-2787

**Methods**:
- `predict(X)` - Generate predictions from ensemble
- `get_weights()` - Return ensemble component weights

### Data Acquisition & Processing

#### `check_gaps(data)`
**Purpose**: Analyze gaps/missing values in time series  
**Returns**: Dictionary with gap information  
**Location**: Line 147

#### `fill_gaps(data)`
**Purpose**: Interpolate missing values in time series  
**Returns**: DataFrame with filled gaps  
**Location**: Line 158

#### `remove_large_gaps(df, col, gap_threshold=6)`
**Purpose**: Remove rows with gaps larger than threshold  
**Parameters**: 
- `col`: Column to check
- `gap_threshold`: Hours of missing data allowed  
**Returns**: Cleaned DataFrame  
**Location**: Line 172

#### `get_historical_weather_api(data, plot)`
**Purpose**: Fetch historical weather from open-meteo API  
**Parameters**: 
- `data`: DataFrame with timestamps (index)
- `plot`: Plot object with GPS coordinates  
**Returns**: DataFrame with weather features  
**Features**: Temperature, humidity, rain, cloudcover, radiation, wind, soil temp/moisture, ET0  
**Location**: Line 188

#### `only_get_historical_weather_api(start, end, plot)`
**Purpose**: Fetch weather for specific date range without alignment  
**Returns**: DataFrame with weather data  
**Location**: Line 262

#### `get_weather_forecast_api(start_date, end_date, plot, data)`
**Purpose**: Fetch weather forecast for future dates  
**Parameters**:
- `plot`: Plot with GPS info
- `data`: Historical data (for validation)  
**Returns**: DataFrame with forecasted weather  
**Location**: Line 312

#### `convert_cols(data)`
**Purpose**: Convert DataFrame columns to float64 dtype  
**Returns**: DataFrame with numeric columns  
**Location**: Line 375

#### `resample(d)`
**Purpose**: Resample time series to standard interval  
**Location**: Line 386

#### `soil_tension_to_volumetric_water_content(soil_tension, soil_water_retention_curve)`
**Purpose**: Convert soil tension (kPa) to volumetric water content (%)  
**Method**: Linear interpolation from retention curve  
**Returns**: Volumetric water content value  
**Location**: Line 391

#### `soil_tension_to_volumetric_water_content_log(soil_tension, soil_water_retention_curve)`
**Purpose**: Convert using logarithmic interpolation  
**Returns**: Volumetric water content value  
**Location**: Line 429

#### `soil_tension_to_volumetric_water_content_spline(soil_tension, soil_water_retention_curve)`
**Purpose**: Convert using cubic spline interpolation  
**Returns**: Volumetric water content value  
**Location**: Line 443

#### `add_volumetric_col_to_df(df, col_name, plot)`
**Purpose**: Add volumetric water content column to DataFrame  
**Parameters**:
- `df`: DataFrame with soil tension values
- `col_name`: Column containing tensions
- `plot`: Plot with soil retention curve  
**Returns**: DataFrame with new `volumetric_water_content` column  
**Location**: Line 467

#### `calc_volumetric_water_content_single_value(soil_tension_value, currentPlot)`
**Purpose**: Calculate VWC for single tension value  
**Returns**: Float value  
**Location**: Line 482

#### `align_retention_curve_with_api(data, data_weather_api, currentPlot)`
**Purpose**: Align soil retention curve with weather API data  
**Returns**: Aligned retention curve  
**Location**: Line 496

#### `add_pump_state(data, plot)`
**Purpose**: Add irrigation state column (0=off, 1=on)  
**Returns**: DataFrame with `pump_state` column  
**Location**: Line 514

#### `hours_since_pump_was_turned_on(df)`
**Purpose**: Calculate hours since last irrigation started  
**Returns**: DataFrame with `hours_since_pump` column  
**Location**: Line 535

#### `ensure_json_file(file_path)`
**Purpose**: Create JSON file if not exists  
**Location**: Line 558

#### `include_irrigation_amount(df, plot)`
**Purpose**: Add irrigation amount from history to dataframe  
**Returns**: DataFrame with `irrigation_amount` column  
**Location**: Line 565

### Feature Engineering

#### `create_features(data, plot)`
**Purpose**: Create all engineered features for ML  
**Features created**:
- Rolling means (15-minute window)
- Gradients (rate of change)
- Grouped statistics (hourly, daily)
- Time features (month, day_of_year, minute)
- Weather interactions  
**Returns**: DataFrame with all features  
**Location**: Line 650

#### `normalize(data)`
**Purpose**: Min-max normalize features to [0,1]  
**Returns**: Normalized DataFrame  
**Location**: Line 734

### Data Splitting & Preparation

#### `split_data_by_date(data, split_date)`
**Purpose**: Split data at specific date boundary  
**Returns**: (train_df, test_df)  
**Location**: Line 760

#### `split_by_ratio(data, test_size_percent)`
**Purpose**: Split data by percentage ratio  
**Parameters**: `test_size_percent` - Percent to allocate to test  
**Returns**: (train_df, test_df)  
**Location**: Line 765

#### `delete_nonconsecutive_rows(df, column_name, min_consecutive)`
**Purpose**: Remove rows with gaps between consecutive values  
**Location**: Line 777

#### `create_split_tuples(df, indices_to_omit)`
**Purpose**: Create train/test split indices avoiding specific rows  
**Returns**: List of train/test index tuples  
**Location**: Line 801

#### `split_dataframe(df, index_ranges)`
**Purpose**: Split dataframe into multiple parts by ranges  
**Location**: Line 826

#### `split_sub_dfs(data, data_plot)`
**Purpose**: Divide data into sub-datasets by irrigation events  
**Returns**: List of sub-dataframes  
**Location**: Line 837

#### `format_begin_end(sub_dfs)`
**Purpose**: Format time ranges of sub-dataframes  
**Location**: Line 896

#### `combine_dfs(cut_sub_dfs)`
**Purpose**: Concatenate list of sub-dataframes  
**Returns**: Single combined DataFrame  
**Location**: Line 938

#### `prepare_data(plot)`
**Purpose**: Main data preparation pipeline (orchestration)  
**Returns**: Cleaned and split training data  
**Location**: Line 957

#### `prepare_data_for_cnn(data, target_variable)`
**Purpose**: Scale and split data for CNN input  
**Returns**: (X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_train_cnn, X_test_cnn, scaler)  
**Location**: Line 1651

#### `prepare_data_for_cnn2(data, target_variable, test_size=0.2, val_size=0.2, random_state=42)`
**Purpose**: Prepare data with train/val/test splits (improved version)  
**Returns**: Full tuple with 12 elements (train, val, test sets in multiple formats)  
**Location**: Line 1681

#### `prepare_future_values(scaler, new_data, X_train_c)`
**Purpose**: Scale future feature data using fitted scaler  
**Returns**: Scaled future data  
**Location**: Line 2211

#### `prepare_lstm_data(data)`
**Purpose**: Reshape data for LSTM input  
**Returns**: Reshaped array  
**Location**: Line 1741

### Model Training

#### `create_and_compare_model_reg(train)`
**Purpose**: Train and compare 19+ regression models using PyCaret  
**Parameters**: `train` - Training dataframe  
**Returns**: (experiment_obj, list_of_best_models)  
**Models compared**: LinearRegression, Ridge, Lasso, ElasticNet, KNN, SVM, DecisionTree, RandomForest, XGBoost, CatBoost, etc.  
**Location**: Line 1108

#### `save_models(plot_name, exp, best, path_to_save)`
**Purpose**: Save trained models to disk  
**Parameters**:
- `exp`: PyCaret experiment object
- `best`: Model or list of models
- `path_to_save`: Directory path  
**Returns**: List of saved file paths  
**Location**: Line 1153

#### `load_models(model_names)`
**Purpose**: Load saved PyCaret models from disk  
**Returns**: List of model objects  
**Location**: Line 1177

#### `evaluate_target_variable(series1, series2, model_name)`
**Purpose**: Calculate metrics (MAE, RMSE, MPE) between predicted and actual  
**Returns**: Dictionary with metric values  
**Location**: Line 1186

#### `evaluate_results_and_choose_best(results_for_one_df, best_for_one_df, pycaret_format=True)`
**Purpose**: Select single best model by R2 score  
**Returns**: Best model object  
**Location**: Line 1231

#### `evaluate_results_and_choose_top_n(results_for_one_df, best_for_one_df, top_n=3, pycaret_format=True)`
**Purpose**: Select top N best models for ensemble  
**Returns**: List of top N models  
**Location**: Line 1248

#### `evaluate_against_testset(currentPlot, test, exp, best)`
**Purpose**: Evaluate model performance on test set  
**Returns**: (best_evaluated_models, results_dict)  
**Location**: Line 1335

#### `train_best(best_model, data)`
**Purpose**: Retrain best model(s) on full dataset (no test split)  
**Returns**: (trained_model(s), experiment_obj)  
**Location**: Line 1369

#### `tune_one_model(exp, best)`
**Purpose**: Hyperparameter tuning for single best model  
**Uses**: PyCaret `tune_one_model()` internally  
**Returns**: Tuned model  
**Location**: Line 2431

#### `tune_models(exp, best)`
**Purpose**: Hyperparameter tuning for multiple models  
**Uses**: PyCaret `tune_models()` internally  
**Returns**: List of tuned models  
**Location**: Line 2449

### Neural Network Models

#### `create_nn_model(hp, shape)`
**Purpose**: Build dense neural network with Keras  
**Architecture**:
- Dense layers with tunable units (32-256)
- Optional 2nd and 3rd hidden layers
- ReLU or tanh activation
- Single output for regression  
**Hyperparameters**: Units, activation, optimizer, learning rate  
**Returns**: Compiled Keras model  
**Location**: Line 1409

#### `create_cnn_model(hp, shape)`
**Purpose**: Build convolutional neural network  
**Architecture**:
- Conv1D layers (tunable filters/kernels)
- MaxPooling1D
- Dense output layer  
**Returns**: Compiled CNN model  
**Location**: Line 1459

#### `create_rnn_model(hp, shape)`
**Purpose**: Build recurrent neural network  
**Architecture**:
- SimpleRNN layers with dropout
- Dense output layer  
**Returns**: Compiled RNN model  
**Location**: Line 1498

#### `create_gru_model(hp, shape)`
**Purpose**: Build GRU (Gated Recurrent Unit) network  
**Architecture**:
- GRU layers with dropout
- Dense output layer  
**Returns**: Compiled GRU model  
**Location**: Line 1546

#### `create_lstm_model(hp, shape)`
**Purpose**: Build LSTM (Long Short-Term Memory) network  
**Architecture**:
- LSTM layers (bidirectional option)
- Dropout regularization
- Dense output layer  
**Returns**: Compiled LSTM model  
**Location**: Line 1593

#### `train_nn_models(X_train, X_val, y_train, y_val, X_train_scaled, X_val_scaled, X_train_cnn, X_val_cnn, plot_name)`
**Purpose**: Train all 5 NN architectures (NN, CNN, RNN, GRU, LSTM)  
**Training parameters**: 50 epochs, batch_size=32, early stopping  
**Returns**: List of trained models  
**Location**: Line 1750

#### `default_hp_nn()`
**Purpose**: Create default hyperparameters for NN  
**Returns**: HyperParameters object with fixed values  
**Location**: Line 1727

#### `save_models_nn(plot_name, nn_models, path_to_save, nn_hps=None)`
**Purpose**: Save NN models and hyperparameters to disk  
**Location**: Line 1968

#### `load_models_nn(folder_path)`
**Purpose**: Load saved NN models from disk  
**Returns**: List of model objects  
**Location**: Line 2023

#### `evaluate_against_testset_nn(currentPlot, nn_models, X_test_scaled, y_test)`
**Purpose**: Evaluate all NN models on test set  
**Metrics**: R2, MAE, MPE  
**Returns**: (best_evaluated_models, results_dict)  
**Location**: Line 2058

#### `train_best_nn(best_eval, data, plot_name)`
**Purpose**: Retrain best NN(s) on full dataset  
**Returns**: (trained_model(s), X_train_scaled, X_val_scaled, y_train, y_val)  
**Location**: Line 2128

#### `tune_model_nn(X_train_scaled, y_train, X_val_scaled, y_val, best_model_nn)`
**Purpose**: Hyperparameter tuning for NN using Hyperband  
**Tuning parameters**: Units, dropout, learning rate  
**Returns**: (tuned_model, best_hyperparameters)  
**Location**: Line 2466

### Ensemble Methods

#### `create_and_compare_ensemble(plot_name, exp, tuned_best_models)`
**Purpose**: Create ensemble of top 3 models (stacking)  
**Ensemble methods**: Stacking, blending, voting  
**Returns**: Ensemble model object  
**Location**: Line 2595

#### `compare_nn_ensembles(models, hps, X_train, y_train, X_val, y_val, metric="r2", bagging_rounds=5, stacking_folds=5, verbose=False)`
**Purpose**: Compare bagging vs stacking for NN ensemble  
**Returns**: Dictionary with {best_predictor, best_name, scores}  
**Location**: Line 2788

### Model Evaluation & Metrics

#### `get_r2(exp, model)`
**Purpose**: Get R² score from PyCaret experiment  
**Returns**: Float R² value  
**Location**: Line 2542

#### `get_r2_manual(model, X_test, y_test)`
**Purpose**: Calculate R² score manually  
**Returns**: Float R² value  
**Location**: Line 2551

#### `smape(y_true, y_pred)`
**Purpose**: Symmetric Mean Absolute Percentage Error  
**Returns**: Float SMAPE score  
**Location**: Line 2235

#### `quantile_loss(y_true, y_pred, q)`
**Purpose**: Quantile loss at specific quantile  
**Returns**: Float loss value  
**Location**: Line 2239

#### `evaluate_target_variable_nd(series1, series2, quantiles=[0.2, 0.4, 0.6, 0.8])`
**Purpose**: Evaluate using multiple quantile losses  
**Returns**: Dictionary with quantile loss values  
**Location**: Line 2244

#### `compare_models_on_test(nn_models, ZZ, Z_cnn)`
**Purpose**: Compare multiple NN models on test set  
**Returns**: Comparison results dictionary  
**Location**: Line 2273

#### `eval_approach(results, results_nn, metrics='mae')`
**Purpose**: Decide between PyCaret vs NN by single metric  
**Returns**: (best_index, use_pycaret_boolean)  
**Location**: Line 2315

#### `eval_approach_mix(results_pycaret, results_nn, weights=None)`
**Purpose**: Decide by weighted combination of MAE, MPE, R2  
**Returns**: (best_index, use_pycaret_boolean)  
**Location**: Line 2337

### Prediction Generation

#### `create_future_values(data, plot)`
**Purpose**: Create feature values for future dates (7+ days)  
**Parameters**:
- `data`: Historical data with all features
- `plot`: Plot with forecast horizon  
**Returns**: DataFrame with future features  
**Location**: Line 1288

#### `generate_predictions(best, exp, features)`
**Purpose**: Generate predictions from PyCaret model  
**Parameters**:
- `best`: Trained PyCaret model
- `exp`: Experiment object (pipeline)
- `features`: Feature dataframe for prediction  
**Returns**: DataFrame with predictions  
**Location**: Line 2988

#### `generate_predictions_nn(best_model_nn, features, start, end)`
**Purpose**: Generate predictions from NN model  
**Returns**: DataFrame with predictions  
**Location**: Line 2998

#### `compare_train_predictions_cols(train, future_features)`
**Purpose**: Align future feature columns with training columns  
**Returns**: DataFrame with aligned columns  
**Location**: Line 2398

#### `plot_history_png(history, filename, y_max=2.0)`
**Purpose**: Plot training history and save as image  
**Returns**: PNG file saved  
**Location**: Line 2092

#### `save_history_json(history, filename)`
**Purpose**: Save training history as JSON  
**Location**: Line 2117

### Post-Processing & Actuation

#### `align_with_latest_sensor_values(plot)`
**Purpose**: Align predictions with latest sensor readings  
**Location**: Line 3051

#### `calc_threshold(predictions, col, plot)`
**Purpose**: Calculate when predictions cross irrigation threshold  
**Parameters**:
- `predictions`: DataFrame with soil moisture predictions
- `col`: Column to analyze
- `plot`: Plot with threshold settings  
**Returns**: Timestamp when threshold will be crossed  
**Location**: Line 3064

#### `quadratic_weights(length)`
**Purpose**: Create quadratic weighting for predictions  
**Returns**: List of weights  
**Location**: Line 3029

#### `exponential_weights(length)`
**Purpose**: Create exponential weighting for predictions  
**Returns**: List of weights  
**Location**: Line 3040

#### `predict_with_updated_data(plot)`
**Purpose**: Generate new predictions using latest data  
**Returns**: (current_soil_tension, threshold_timestamp, predictions_df)  
**Location**: Line 3082

### Main Pipeline

#### `data_pipeline(plot)`
**Purpose**: Orchestrate entire data acquisition and preparation  
**Steps**:
1. Fetch sensor data from WaziGate API
2. Fetch weather (historical + forecast)
3. Align and clean data
4. Create features
5. Handle irrigation events
6. Split into train/test  
**Returns**: (train_df, test_df, X_train, X_val, X_test, y_train, y_val, y_test, scaled_versions, scaler)  
**Location**: Line 3125

#### `main(plot) -> int`
**Purpose**: Main training orchestration function  
**Steps**:
1. Data pipeline (prepare data)
2. PyCaret model comparison + evaluation
3. NN model training + evaluation
4. Decision: which approach is best
5. Hyperparameter tuning (IN SUBPROCESS)
6. Ensemble creation (IN SUBPROCESS)
7. Generate predictions
8. Calculate irrigation threshold
9. Return results  
**Returns**: (current_soil_tension, threshold_timestamp, predictions_df)  
**Location**: Line 3142

---

## Subprocess Management (subprocess_manager.py) ⚠️

### Classes

#### `SubprocessManager`
**Purpose**: Manages subprocess execution for memory-intensive operations  
**Location**: Lines 24-430

**Key Attributes**:
- `max_memory_percent`: Kill subprocess if RAM exceeds this %
- `timeout_seconds`: Maximum execution time
- `temp_dir`: Directory for subprocess scripts and data

**Methods**:

##### `run_tuning_pycaret(exp_pickle_path, best_models_pickle_path, result_path, plot_name, ensemble=False, tune_grids=None)`
**Purpose**: Run PyCaret tuning in subprocess  
**Parameters**:
- `exp_pickle_path`: Path to saved experiment object
- `best_models_pickle_path`: Path to saved models
- `result_path`: Where to save results
- `plot_name`: Name for logging
- `ensemble`: Whether tuning for ensemble  
**Returns**: Path to result pickle file  
**Memory Savings**: 20-50% reduction in peak usage

##### `run_ensemble_pycaret(exp_pickle_path, best_models_pickle_path, result_path, plot_name)`
**Purpose**: Run ensemble creation in subprocess  
**Returns**: Path to ensemble model pickle file

##### `run_tuning_nn(model_pickle_path, X_train_pickle_path, y_train_pickle_path, X_val_pickle_path, y_val_pickle_path, result_path, plot_name)`
**Purpose**: Run NN tuning in subprocess  
**Returns**: (tuned_model_path, best_hp_path)

##### `run_ensemble_nn(models_pickle_path, hps_pickle_path, X_train_pickle_path, y_train_pickle_path, X_val_pickle_path, y_val_pickle_path, result_path, plot_name)`
**Purpose**: Run NN ensemble in subprocess  
**Returns**: Path to ensemble result pickle file

##### `_monitor_process(plot_name, check_interval=5)`
**Purpose**: Monitor subprocess for timeout and errors  
**Parameters**:
- `plot_name`: For logging
- `check_interval`: How often to check (seconds)  
**Returns**: (stdout, stderr)

##### `cleanup_old_temps(max_age_hours=24)`
**Purpose**: Delete old temporary files  
**Parameters**: `max_age_hours` - Delete files older than this

##### `_cleanup(script_path)`
**Purpose**: Remove temporary subprocess script  

### Functions

#### `run_tuning_with_subprocess(exp, best_model, plot_name, ensemble=False)`
**Purpose**: Convenience wrapper for PyCaret tuning  
**Returns**: Tuned model(s)  
**Location**: Line 431

#### `run_ensemble_with_subprocess(exp, best_model, plot_name)`
**Purpose**: Convenience wrapper for ensemble creation  
**Returns**: Ensemble model  
**Location**: Line 477

---

## Plot Management (plot.py, plot_manager.py)

### Plot Class (plot.py)

#### `Plot`
**Purpose**: Data model for individual agricultural field  
**Location**: Lines 14-472

**Key Attributes**:
- `id`, `tab_number`, `user_given_name`: Plot identification
- `device_and_sensor_ids_moisture`, `_temp`, `_flow`: Sensor references
- `gps_info`: GPS coordinates
- `sensor_kind`: "tension" or "volumetric"
- `threshold`, `irrigation_amount`, `look_ahead_time`: Irrigation settings
- `start_date`, `period`, `train_period_days`: Training parameters
- `soil_type`, `permanent_wilting_point`, `field_capacity_*`: Soil properties
- `currently_training`, `training_finished`: State flags
- `predictions`, `threshold_timestamp`: Prediction results
- `best_model`, `best_exp`: Trained model and experiment

**Methods**:
- `read_config()` - Load configuration from JSON
- `write_config()` - Save configuration to JSON
- `printPlotNumber()` - Debug output
- `setState()` - Set training state
- `check_threads()` - Monitor background threads

### Plot Manager Functions (plot_manager.py)

#### `readFiles()`
**Purpose**: List all config files in config/ directory  
**Returns**: Sorted list of filenames  
**Location**: Line 17

#### `setPlot(plot_nr_tab)`
**Purpose**: Change current active plot  
**Parameters**: `plot_nr_tab` - Tab number in UI  
**Returns**: Config path for new plot  
**Location**: Line 27

#### `loadPlots()`
**Purpose**: Load all plots from config files at startup  
**Returns**: Number of plots loaded  
**Location**: Line 53

#### `addPlot(tabNumber)`
**Purpose**: Create new plot  
**Returns**: (new_tab_number, config_file_path)  
**Location**: Line 73

#### `removePlot(plot_nr_to_be_removed)`
**Purpose**: Delete plot by tab number  
**Location**: Line 103

#### `getPlots()`
**Purpose**: Get all plot objects  
**Returns**: Dictionary of {tab_num: Plot}  
**Location**: Line 143

#### `getCurrentConfig()`
**Purpose**: Get config of current active plot  
**Returns**: Config dictionary  
**Location**: Line 146

#### `getCurrentPlot()`
**Purpose**: Get current Plot object  
**Returns**: Plot instance  
**Location**: Line 149

#### `getCurrentPlotNumberWithId(currentPlot)`
**Purpose**: Find tab number given plot object  
**Returns**: Tab number  
**Location**: Line 161

#### `getCurrentPlotWithId(passed_id)`
**Purpose**: Get Plot object by plot ID  
**Returns**: Plot instance  
**Location**: Line 170

#### `removePlotWithId(passed_id)`
**Purpose**: Delete plot by ID  
**Location**: Line 176

#### `setCurrentConfig(path)`
**Purpose**: Set current config file path  
**Location**: Line 186

---

## Threading (training_thread.py, prediction_thread.py)

### Training Thread (training_thread.py)

#### `TrainingThread(threading.Thread)`
**Purpose**: Background thread for scheduled model training  
**Location**: Lines 16-113

**Methods**:

##### `__init__(self, plot, startTrainingNow, name=None)`
**Parameters**:
- `plot`: Plot object to train
- `startTrainingNow`: Boolean to train immediately
- `name`: Thread name

##### `time_until_noon(train_period_days)`
**Purpose**: Calculate seconds until next scheduled training time  
**Returns**: Seconds to sleep  

##### `calculate_retrain_interval(current_data_days)`
**Purpose**: Calculate adaptive training interval  
**Formula**: Logarithmic growth from 1 day to 30 days max  
**Returns**: Integer days until next training

##### `run()`
**Purpose**: Main thread loop  
**Behavior**:
1. Wait until scheduled time
2. Call `create_model.main()` for training
3. Save results to pickle
4. Trigger actuation if flow meters exist
5. Start prediction thread
6. Repeat with new interval

##### `start(currentPlot)`
**Purpose**: Module-level function to start training thread  
**Location**: Line 114

---

### Prediction Thread (prediction_thread.py)

#### `PredictionThread(threading.Thread)`
**Purpose**: Background thread for scheduled predictions  
**Location**: Lines 14-89

**Methods**:

##### `__init__(plot, name=None)`
**Parameters**: `plot` - Plot object for predictions

##### `time_until_n_hours(hours)`
**Purpose**: Calculate sleep time until next prediction  
**Returns**: Seconds to sleep

##### `run()`
**Purpose**: Main thread loop  
**Behavior**:
1. Wait for `predict_period_hours`
2. Call `create_model.predict_with_updated_data()`
3. Trigger actuation
4. Repeat

##### `stop()`
**Purpose**: Signal thread to stop

##### `start(currentPlot)`
**Purpose**: Module-level function to start prediction thread  
**Location**: Line 90

---

## Irrigation Control (actuation.py)

### Functions

#### `get_max_min(df, target_col='smoothed_values')`
**Purpose**: Find maximum/minimum values in predictions  
**Returns**: (min_index, max_index, min_value, max_value)  
**Location**: Line 22

#### `find_next_occurrences(df, column, threshold, timeSpanOverThreshold)`
**Purpose**: Find when predictions cross threshold  
**Parameters**:
- `df`: Predictions dataframe
- `column`: Column to check
- `threshold`: Value threshold
- `timeSpanOverThreshold`: Hours to maintain crossing  
**Returns**: List of times when threshold is crossed  
**Location**: Line 40

#### `read_data_from_file(filename)`
**Purpose**: Read irrigation history from JSON file  
**Returns**: Dictionary of irrigation records  
**Location**: Line 86

#### `save_data_to_file(filename, data)`
**Purpose**: Save irrigation history to JSON  
**Location**: Line 94

#### `add_record(data, timestamp, amount, status="not confirmed")`
**Purpose**: Add irrigation event to history  
**Returns**: Updated data dictionary  
**Location**: Line 102

#### `round_to_nearest_10_minutes(dt)`
**Purpose**: Round timestamp to nearest 10-minute interval  
**Returns**: Rounded datetime  
**Location**: Line 113

#### `save_irrigation_time(amount, plotid, status="not confirmed") -> int`
**Purpose**: Save irrigation event with timestamp  
**Returns**: Number of irrigation events  
**Location**: Line 124

#### `update_irrigation_status(plot, status="not_confirmed")`
**Purpose**: Update status of last irrigation (confirmed/failed)  
**Location**: Line 147

#### `verify_irrigation(plot, amount)`
**Purpose**: Check if actual irrigation occurred via flow meter  
**Verification**: Compare expected vs actual water flow  
**Retries**: Up to `Irrigation_retries` times  
**Returns**: Boolean success  
**Location**: Line 161

#### `irrigate_amount(plot, amount=0)`
**Purpose**: Send irrigation command to WaziGate  
**Parameters**: 
- `plot`: Plot object
- `amount`: Liters to dispense (0=use default)  
**Location**: Line 220

#### `main_old(currentSoilTension, threshold_timestamp, predictions, plot) -> int`
**Purpose**: Old actuation logic (deprecated)  
**Location**: Line 276

#### `main(currentSoilTension, threshold_timestamp, predictions, plot) -> int`
**Purpose**: Main irrigation decision and execution  
**Logic**:
1. Check if threshold will be crossed in look-ahead window
2. If yes, trigger irrigation
3. Verify with flow meter
4. Update status
5. Log event  
**Returns**: Status code  
**Location**: Line 318

---

## Utilities (utils.py)

### TimeUtils Class

#### `TimeUtils`
**Purpose**: Timezone utilities and conversions  
**Location**: Lines 10-41

**Methods**:

##### `get_timezone_offset(timezone_str)` - static
**Purpose**: Get UTC offset in hours for timezone  
**Returns**: Float offset in hours

##### `get_timezone(latitude_str, longitude_str)` - static
**Purpose**: Get timezone string from GPS coordinates  
**Uses**: TimezoneFinder library  
**Returns**: Timezone string (e.g., "Africa/Nairobi")

### NetworkUtils Class

#### `NetworkUtils`
**Purpose**: Network and API utilities  
**Location**: Lines 42-110

**Class Attributes**:
- `Env`: Environment variables dictionary
- `ApiUrl`: WaziGate API base URL
- `Proxy`: Proxy URL (optional)
- `Token`: Authentication token

**Methods**:

##### `get_env()` - classmethod
**Purpose**: Load environment variables from .env file  

##### `get_token()` - classmethod
**Purpose**: Get authentication token for remote gateways  
**Location**: Lines 73-110

---

## HTTP Routing (usock.py)

### Router Functions

#### `routerGET(path, func)`
**Purpose**: Register GET route handler  
**Parameters**:
- `path`: URL path pattern
- `func`: Handler function  
**Location**: Line 28

#### `routerPOST(path, func)`
**Purpose**: Register POST route handler  
**Location**: Line 35

#### `routerPUT(path, func)`
**Purpose**: Register PUT route handler  
**Location**: Line 42

#### `routerDELETE(path, func)`
**Purpose**: Register DELETE route handler  
**Location**: Line 49

### HTTP Handler

#### `HTTPHandler(BaseHTTPRequestHandler)`
**Purpose**: Custom HTTP request handler  
**Location**: Lines 58-161

**Methods**:
- `do_GET()` - Handle GET requests
- `do_POST()` - Handle POST requests
- `do_PUT()` - Handle PUT requests
- `do_DELETE()` - Handle DELETE requests

### Server Functions

#### `start()`
**Purpose**: Start HTTP server on port 3030  
**Location**: Line 162

#### `start_with_recovery()`
**Purpose**: Start server with automatic restart on failure  
**Location**: Line 202

---

## Data Models & Constants

### Global Variables (create_model.py)

```python
RollingMeanWindowData = 15              # Smoothing window size (minutes)
RollingMeanWindowGrouped = 5            # Grouped smoothing window
Sample_rate = 60                        # Minutes between sensor samples
Forcast_horizon = 5                     # Days into future
Verbose_logging = True/False            # Debug flag
SkipDataPreprocessing = False           # Use debug CSV instead of API
SkipTraning = False                     # Use debug predictions
Perform_training = True                 # Execute training vs load from disk
Currently_active = False                # Global training lock
```

### Configuration Structure (JSON format)

```json
{
  "user_given_name": "Plot 1",
  "gps_info": {
    "lattitude": 0.35,
    "longitude": 32.5
  },
  "sensors": {
    "moisture": [{"device_id": "...", "sensor_id": "..."}],
    "temperature": [{"device_id": "...", "sensor_id": "..."}],
    "flow_meter": [{"device_id": "...", "sensor_id": "..."}]
  },
  "irrigation": {
    "threshold": 30,
    "amount": 100,
    "look_ahead": 24
  },
  "soil": {
    "type": "loam",
    "wilting_point": 40,
    "field_capacity_upper": 30,
    "field_capacity_lower": 10,
    "saturation": 0,
    "water_retention_curve": [[0, 0.45], [50, 0.30], ...]
  },
  "training": {
    "start_date": "2024-01-01",
    "period": 365,
    "interval": 1,
    "ensemble": true
  }
}
```

---

## Global Actuation Constants (actuation.py)

```python
OverThresholdAllowed = 1.2              # 20% tolerance for threshold
Irrigation_confirmation_sec = 10800     # 3 hours until verification
Irrigation_retries = 0                  # Number of retry attempts
```

---

## Return Value Conventions

### API Response Format
```json
{
  "status": "success" | "error",
  "message": "Description",
  "data": {...}
}
```

### Status Codes
- **200**: Success
- **400**: Bad request
- **404**: Not found
- **500**: Internal error

---

## Error Handling

### Try-Except Patterns
- Subprocess failures fall back to original model
- API timeouts retry with exponential backoff
- Missing sensor data is interpolated
- JSON parse errors return empty defaults

### Graceful Degradation
- If ensemble fails → use tuned single model
- If tuning fails → use comparison results
- If NN fails → use PyCaret
- If model fails → use dummy predictor

---

## Performance Optimization

### Memory Management
- Subprocess isolation for tuning/ensemble
- Generator functions for large datasets
- Garbage collection between major operations
- Pickle for serialization (vs JSON for large arrays)

### Speed Optimization
- PyCaret uses n_jobs=1 (single core)
- Early stopping for NN training
- Vectorized numpy operations
- Cached computed features

### Parallelization
- Multi-threaded: training + prediction threads
- Single-process tuning/ensemble (in subprocesses)
- No multiprocessing for model training (RAM constraint)

---

**Last Updated**: March 20, 2026  
**Version**: 1.0 with Subprocess Optimization
