"""Immutable configuration constants shared across the create_model package."""

# Rolling mean window
RollingMeanWindowData = 15
RollingMeanWindowGrouped = 5

# Sampling rate of training dataset
Sample_rate = 60

# Forecast horizon TODO: add config or adjust automa !!!!
Forecast_horizon = 5 #days

# Number of resampled rows spanned by one forecast horizon (used as the CV gap below,
# so cross-validation is scored on a true forecast-horizon-ahead split instead of the
# next timestamp, which would be far easier to predict and overstate model quality)
Forecast_horizon_periods = int(Forecast_horizon * 24 * 60 / Sample_rate)

# Created features that are dropped later -> TODO: evaluate this!!!
To_be_dropped = ['minute', 'Timestamp', 'gradient', 
                 'grouped_soil', 'grouped_soil_temp', 
                 'Winddirection', 'month', 'day_of_year', 
                 'date']

# Mapping to identify models
Model_mapping = {
    'LinearRegression': 'lr',
    'Lasso': 'lasso',
    'Ridge': 'ridge',
    'ElasticNet': 'en',
    'Lars': 'lar',
    'LassoLars': 'llar',
    'OrthogonalMatchingPursuit': 'omp',
    'BayesianRidge': 'br',
    'ARDRegression': 'ard',
    'PassiveAggressiveRegressor': 'par',
    'RANSACRegressor': 'ransac',
    'TheilSenRegressor': 'tr',
    'HuberRegressor': 'huber',
    'KernelRidge': 'kr',
    'SVR': 'svm',
    'KNeighborsRegressor': 'knn',
    'DecisionTreeRegressor': 'dt',
    'RandomForestRegressor': 'rf',
    'ExtraTreesRegressor': 'et',
    'AdaBoostRegressor': 'ada',
    'GradientBoostingRegressor': 'gbr',
    'MLPRegressor': 'mlp',
    'XGBRegressor': 'xgboost',
    'LGBMRegressor': 'lightgbm',
    'CatBoostRegressor': 'catboost',
    'DummyRegressor': 'dummy'
}

# Wait time for resources to be released (recheck after busy in training or prediction for other plots)
Resource_wait_time_seconds = 300    # seconds

# Restrict memory usage: stop training before the kernel OOM-killer kills the whole
# process - on a 4GB RPi the 99% DEBUG setting never fires in time (was 75 originally)
Memory_limit_percent = 85
