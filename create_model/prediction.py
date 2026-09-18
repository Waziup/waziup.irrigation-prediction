"""Forecast feature assembly, inference, and post-processing."""

import pandas as pd
import numpy as np

from .constants import *
from .features import Weather_derived_feature_cols, add_weather_derived_features
from .nn_architectures import adapt_X_for_model, safe_model_name
from .weather import get_weather_forecast_api, only_get_historical_weather_api
from weather_quality import deduplicate_weather_frame


# Create future value testset for prediction
def create_future_values(data, plot):
    # Create ranges and dataframe with timestamps
    start = data.index[0]
    print("start: ", start)
    train_end = data.index[-1]  # + timedelta(minutes=sample_rate)
    print("train end before adding: ", train_end)
    # Build future features with the same per-farm cadence and horizon exposed
    # to orchestration and the decision engine.
    horizon_days = float(
        getattr(plot, "forecast_horizon_days", Forecast_horizon))
    interval_minutes = float(
        getattr(plot, "forecast_interval_minutes", Sample_rate))
    if (not np.isfinite(horizon_days) or not np.isfinite(interval_minutes)
            or horizon_days <= 0 or interval_minutes <= 0):
        raise ValueError(
            "forecast_horizon_days and forecast_interval_minutes must be positive")
    # Training resampling and antecedent feature windows use Sample_rate.
    # Until interval-total resampling is defined end-to-end, do not interpolate
    # Rain/ET0 or silently change the temporal meaning of trained inputs.
    if interval_minutes != Sample_rate:
        raise ValueError("Forecast cadence must match the training sample rate")
    interval_minutes = int(interval_minutes)
    if (not isinstance(data.index, pd.DatetimeIndex) or data.index.hasnans
            or not data.index.is_unique or not data.index.is_monotonic_increasing
            or not (data.index.to_series().diff().dropna()
                    == pd.Timedelta(minutes=interval_minutes)).all()):
        raise ValueError("Training history must have a regular training cadence")
    end = train_end + pd.Timedelta(days=horizon_days)
    print("end after adding: ", end, "\n")
    all_dates = pd.date_range(start=train_end, end=end,
                              freq=str(interval_minutes)+'min')
    print("all dates: ", all_dates, "\n")

    # Fetch data from weather API
    if plot.load_data_from_csv:
        data_weather_api_cut = only_get_historical_weather_api(
            train_end, end, plot)
    else:
        data_weather_api_cut = get_weather_forecast_api(
            train_end, end, plot, data)
    if not isinstance(data_weather_api_cut.index, pd.DatetimeIndex):
        raise ValueError("Forecast weather requires a datetime index")
    data_weather_api_cut = deduplicate_weather_frame(data_weather_api_cut)
    in_window = data_weather_api_cut.loc[
        (data_weather_api_cut.index >= train_end)
        & (data_weather_api_cut.index <= end)]
    if not in_window.index.isin(all_dates).all():
        raise ValueError("Weather cadence must match the model timestamp grid")

    # Create features and merge data from weather API
    # Keep precisely the model grid, never the union with off-grid provider rows.
    new_data = data_weather_api_cut.reindex(all_dates).copy()
    required = ['Temperature', 'Humidity', 'Rain', 'Et0_evapotranspiration',
                'Soil_temperature_7-28']
    if not set(required).issubset(new_data.columns):
        raise ValueError("Forecast weather is missing required model inputs")
    numeric = new_data.apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Forecast weather must provide finite inputs at every model timestamp")
    new_data = numeric.rename_axis('Timestamp')
    new_data.reset_index(inplace=True)  # Reset the index
    new_data.rename(columns={'index': 'Timestamp'}, inplace=True)

    # dates
    new_data['hour'] = [i.hour for i in new_data['Timestamp']]
    # minute is not important
    new_data['minute'] = [i.minute for i in new_data['Timestamp']]
    new_data['date'] = [i.day for i in new_data['Timestamp']]
    new_data['month'] = [i.month for i in new_data['Timestamp']]
    new_data['day_of_year'] = [i.dayofyear for i in new_data['Timestamp']]

    # make up some other data from weatherAPI
    # new_data['rolling_mean_grouped_soil_vol'] = new_data['Soil_moisture_0-7'] #Approach is not any more used
    # TODO: calculate/calibrate diviation for better alignment
    new_data['rolling_mean_grouped_soil_temp'] = new_data['Soil_temperature_7-28']

    # also include pump_state or irrigation_amount depending on config, set to zero as we want to assume the behavior without watering
    if "DeviceAndSensorIdsFlow" in plot.config:
        new_data = new_data.assign(irrigation_amount=0)
    else:
        new_data = new_data.assign(pump_state=0)

    # Compute the same weather-derived features as in training (create_features).
    # The rolling windows need antecedent history: prepend the last 7 days of observed
    # weather from the training data so the windows at the forecast start see the real
    # recent conditions instead of an empty warmup, then cut the history rows off again.
    feature_inputs = ['Temperature', 'Humidity',
                      'Rain', 'Et0_evapotranspiration', 'hour']
    # Give water_balance_7d the measured irrigation history where a flow meter exists
    # (forecast rows themselves carry irrigation_amount=0, assigned above)
    if 'irrigation_amount' in new_data.columns and 'irrigation_amount' in data.columns:
        feature_inputs.append('irrigation_amount')
    first_forecast_ts = new_data['Timestamp'].iloc[0]
    history_hours = max(168, horizon_days * 24)
    history_rows = max(1, int(history_hours * 60 / interval_minutes))
    history = data.loc[data.index < first_forecast_ts,
                       feature_inputs].tail(history_rows)
    combined = pd.concat(
        [history.reset_index(drop=True),
         new_data[feature_inputs].reset_index(drop=True)],
        axis=0, ignore_index=True
    )
    combined = add_weather_derived_features(combined)
    new_data[Weather_derived_feature_cols] = combined[Weather_derived_feature_cols].tail(
        len(new_data)).to_numpy()

    return new_data


# Compare dataframes cols to be sure that they match, otherwise drop
def compare_train_predictions_cols(train, future_features):
    """Match the ordered training inputs without fabricating missing values."""
    if not train.columns.is_unique or not future_features.columns.is_unique:
        raise ValueError("Training and prediction feature columns must be unique")
    excluded = set(To_be_dropped) | {"rolling_mean_grouped_soil"}
    expected = [column for column in train.columns if column not in excluded]
    missing = [column for column in expected if column not in future_features.columns]
    if missing:
        raise ValueError(f"Missing prediction feature columns: {missing}")
    aligned = future_features.copy()
    if "Timestamp" in aligned.columns:
        aligned = aligned.set_index("Timestamp")
    elif not isinstance(aligned.index, pd.DatetimeIndex):
        raise ValueError("Prediction features require a Timestamp column or DatetimeIndex")
    # Extra provider fields and historical-only/target columns are not inputs.
    # Selecting by the ordered list also protects array-based scaler/model use.
    return aligned.loc[:, expected].copy()


def _validated_predictions(values, expected_rows):
    """Require one finite, real numeric prediction per input row before clipping."""
    array = np.asarray(values)
    if (array.shape not in ((expected_rows,), (expected_rows, 1))
            or expected_rows == 0 or array.dtype.kind not in 'iuf'
            or not np.isfinite(array).all()):
        raise ValueError("Model outputs must contain one finite numeric prediction per input row")
    return array.reshape(-1)


# Generate prediction with best_model and impute generated future_values
def generate_predictions(best, exp, features):
    # Generate predictions
    predictions = exp.predict_model(best, data=features)
    if not isinstance(predictions, pd.DataFrame) or 'prediction_label' not in predictions:
        raise ValueError("Model outputs require a prediction_label column")
    values = _validated_predictions(predictions['prediction_label'], len(features))
    predictions = predictions.copy()

    # Clip neg predictions to zero
    predictions['prediction_label'] = np.maximum(values, 0)

    return predictions


# Generating predictions with neural network model
def generate_predictions_nn(best_model_nn, features, start, end, interval_minutes=Sample_rate):
    """Generate NN predictions indexed at the configured model cadence."""
    print("Generating predictions with NN model:",
          safe_model_name(best_model_nn))

    # Ensure features is a numpy array
    X_pred = np.asarray(features)

    # Adapt features to model input shape
    X_pred = adapt_X_for_model(best_model_nn, X_pred)

    # Generate predictions
    predictions = best_model_nn.predict(X_pred)
    predictions = _validated_predictions(predictions, len(X_pred))

    # Clip neg predictions to zero
    predictions = np.maximum(predictions, 0)

    # Convert the numpy array to a pandas DataFrame and name the column
    predictions = pd.DataFrame(predictions, columns=['prediction_label'])

    # Add timestamps to the dataframe -> create a DatetimeIndex
    date_range = pd.date_range(
        start=start, end=end, freq=str(interval_minutes)+'min')

    # Ensure the length of date_range matches the DataFrame
    if len(date_range) != len(predictions):
        raise ValueError(
            "Length of date_range does not match length of DataFrame")

    # Replace the index with the new DatetimeIndex
    predictions.index = date_range

    return predictions


# Exponential weighting function, to be used in align_with_latest_sensor_values
def exponential_weights(length):
    """
    Generate exponential weights for blending.
    The weights start high (.5) and decrease exponentially to (.1). (moderate)
    """
    x = np.linspace(0, 1, length)   # Normalized positions
    weights = np.exp(-.5*x) - .5     # Exponential decay

    return weights


# align prediction according to latest sensor values
def align_with_latest_sensor_values(plot):
    # Extract the last actual value
    last_actual_value = plot.data['rolling_mean_grouped_soil'].iloc[-1]
    _validated_predictions([last_actual_value], 1)
    _validated_predictions(plot.predictions['prediction_label'], len(plot.predictions))

    # Generate weights for the prediction range
    weights = exponential_weights(len(plot.predictions))

    # Step 3: Blend the historical and predicted values
    plot.predictions['smoothed_values'] = (
        weights * last_actual_value +
        (1 - weights) * plot.predictions['prediction_label']
    )


# Calculates the time when threshold will be meet, according to predictions
def calc_threshold(predictions, col, plot):
    _validated_predictions(predictions[col], len(predictions))
    threshold = plot.threshold
    strategy = plot.sensor_kind

    # Define comparison logic based on strategy
    comparison_fn = (lambda value, threshold: value >= threshold) if strategy == "tension" else (
        lambda value, threshold: value <= threshold
    )

    # calculate next occurance
    for i in range(len(predictions)):
        if comparison_fn(predictions[col][i], threshold):
            print("Threshold will be reached on",
                  predictions.index[i], "With a value of:", predictions[col][i])
            return predictions.index[i]

    return ""
