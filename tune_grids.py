# Pre-defined hyperparameter grids for regression models in PyCaret. -> they are very shallow grids, for quick tuning on small datasets

# TODO: move this to a json!?

# Tests Summary:
# lr ok
# lasso ok
# ridge ok
# en ok
# lar ok
# llar ok
# omp ok
# br ok
# ard ok
# par ok
# ransac ok
# tr ok
# huber ok
# kr ok
# svm ok
# knn ok
# dt ok
# rf ok
# et ok
# ada ok
# gbr ok
# mlp ok
# xgboost ok
# lightgbm ok
# catboost ok
# dummy ok

PYCARET_REGRESSION_TUNE_GRIDS = {
    # -------------------------------------------------
    # Linear / generalized linear models
    # -------------------------------------------------
    "lr": {  # Linear Regression
        "fit_intercept": [True, False],
    },

    "lasso": {  # Lasso Regression
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        "fit_intercept": [True, False],
        "max_iter": [1000, 5000],
    },

    "ridge": {  # Ridge Regression
        "alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100],
        "fit_intercept": [True, False],
        "solver": ["auto", "svd", "lsqr"],
    },

    "en": {  # Elastic Net
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1],
        "l1_ratio": [0.1, 0.5, 0.9],
        "fit_intercept": [True, False],
        "max_iter": [1000, 5000],
    },

    "lar": {
        # whether to copy X or modify in-place
        "copy_X": [True, False],
        # precision of the computation
        "eps": [1e-6, 1e-5, 1e-4],
        # whether to calculate the intercept
        "fit_intercept": [True, False],
        # whether to compute full path
        "fit_path": [True, False],
        # added to the covariance matrix for stability
        "jitter": [1e-6, 1e-5],
        # number of non-zero coefficients (None = no limit)
        "n_nonzero_coefs": [5, 10, 20, None],
        # whether to use precomputed Gram matrix
        "precompute": [True, False],
        # for reproducibility
        "random_state": [42],
        # verbosity level
        "verbose": [0, 1],        
    },

    "llar": {
        # whether to copy X or modify in-place
        "copy_X": [True, False],
        # precision of the computation
        "eps": [1e-6, 1e-5, 1e-4],
        # whether to calculate the intercept
        "fit_intercept": [True, False],
        # whether to compute full path
        "fit_path": [True, False],
        # added to the covariance matrix for stability
        "jitter": [1e-6, 1e-5],
        # number of non-zero coefficients (None = no limit)
        "n_nonzero_coefs": [5, 10, 20, None],
        # whether to use precomputed Gram matrix
        "precompute": [True, False],
        # for reproducibility
        "random_state": [42],
        # verbosity level
        "verbose": [0, 1],                   
    },

    "omp": {
        # whether to calculate the intercept
        "fit_intercept": [True, False],
        # max number of non-zero coefficients (None = no limit)
        "n_nonzero_coefs": [5, 10, 20, None],
        # whether to use precomputed Gram matrix
        "precompute": [True, False], 
        # tolerance for stopping criterion
        "tol": [1e-6, 1e-5, 1e-4],             
    },

    "br": {  # Bayesian Ridge
        # standard sklearn params:
        "n_iter": [300, 600],
        "alpha_1": [1e-6, 1e-5],
        "alpha_2": [1e-6, 1e-5],
        "lambda_1": [1e-6, 1e-5],
        "lambda_2": [1e-6, 1e-5],
    },

    "ard": {  # Automatic Relevance Determination
        # Less commonly tuned. These are educated guesses:
        "n_iter": [300, 600],
        "alpha_1": [1e-7, 1e-6],
        "alpha_2": [1e-7, 1e-6],
        "lambda_1": [1e-7, 1e-6],
        "lambda_2": [1e-7, 1e-6],
        "threshold_lambda": [1e4, 1e5],
    },

    "par": {  # Passive Aggressive Regressor
        "C": [0.01, 0.1, 1.0],
        "fit_intercept": [True, False],
        "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
        "epsilon": [0.01, 0.1],
    },

    "ransac": {  # Random Sample Consensus
        # RANSAC wraps a base estimator (often LinearRegression) and tuning it is tricky. These are tentative:
        "min_samples": [0.1, 0.25, 0.5],
        "residual_threshold": [1.0, 5.0, 10.0],
        "max_trials": [100, 500],
    },

    "tr": {  # TheilSen Regressor
        # Very expensive for large data
        # Parameters below are educated guesses:
        "max_subpopulation": [10000, 20000],
        "n_subsamples": [None, 200, 500],
        "fit_intercept": [True, False],
    },

    "huber": {  # Huber Regressor
        "epsilon": [1.1, 1.35, 1.5],
        "alpha": [1e-4, 1e-3, 1e-2],
        "fit_intercept": [True, False],
        "max_iter": [100, 500],
    },

    "kr": {  # Kernel Ridge
        # Kernel choice + alpha, gamma. These are fairly generic guesses:
        "alpha": [1e-3, 1e-2, 1e-1, 1],
        "kernel": ["rbf", "poly"],
        "gamma": [None, 0.01, 0.1],     # not used for all kernels
        "degree": [2, 3],               # for poly kernel
    },

    "svm": {  # Support Vector Regression
        "C": [0.1, 1, 10],
        "epsilon": [0.01, 0.1, 0.5],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    },

    "knn": {  # K Neighbors Regressor
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2],  # 1=Manhattan, 2=Euclidean
    },

    # -------------------------------------------------
    # Trees / ensembles
    # -------------------------------------------------
    "dt": {  # Decision Tree Regressor
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [None, "sqrt", "log2"],
    },

    "rf": {  # Random Forest Regressor
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.8],
        "bootstrap": [True, False],
    },

    "et": {  # Extra Trees Regressor
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.8],
    },

    "ada": {  # AdaBoost Regressor
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
        # base_estimator params are trickier – leaving them as default.
    },

    "gbr": {  # Gradient Boosting Regressor
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.6, 0.8, 1.0],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    },

    # -------------------------------------------------
    # Neural nets
    # -------------------------------------------------
    "mlp": {  # MLP Regressor (PyCaret-compatible) , TODO: is redundant, already one that can be tuned in my implementation 
        "hidden_layer_size_0": [50, 100],      # first (and only) hidden layer
        "activation": ["relu", "tanh"],
        "alpha": [1e-4, 1e-3],                # regularization
        "learning_rate": ["constant", "adaptive"],
        "max_iter": [200, 300],                # keep small for Pi
        "random_state": [42],
    },

    # -------------------------------------------------
    # Gradient boosting libs
    # -------------------------------------------------
    "xgboost": {  # XGBRegressor
        "n_estimators": [200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "reg_lambda": [1, 5, 10],
        "gamma": [0, 1, 5],
    },

    "lightgbm": {
        # number of trees – keep small, dataset is tiny
        "n_estimators": [100, 300],
        # small trees: 2^5 = 32 leaves is already plenty
        "num_leaves": [15, 31],
        # learning rate + n_estimators trade-off
        "learning_rate": [0.05, 0.1],
        # regularization via min data per leaf
        "min_child_samples": [10, 20, 40],
        # row subsampling – light shrinkage to fight overfitting
        "subsample": [0.8, 1.0],
        # column subsampling
        "colsample_bytree": [0.8, 1.0],
        # L2 regularization
        "reg_lambda": [0, 1],
    },

    "catboost": {  # CatBoostRegressor
        # Tree structure
        "depth": [4, 6, 8],
        # Learning rate & iterations
        "learning_rate": [0.03, 0.06, 0.1],
        "iterations": [300, 600],
        # Regularization
        "l2_leaf_reg": [1, 3, 5],
        # Number of bins for numerical features
        "border_count": [32, 64, 128],
        # CatBoost-specific bagging intensity
        # 0 = no bagging, >1 = more aggressive, more stochastic
        "bagging_temperature": [0, 0.5, 1.0],
        # Row sampling
        "subsample": [0.8, 1.0],
    },

    # -------------------------------------------------
    # Baselines / misc
    # -------------------------------------------------
    "kr": {  # note: already defined above, but here to emphasize it exists
        "alpha": [1e-3, 1e-2, 1e-1, 1],
        "kernel": ["rbf", "poly"],
        "gamma": [None, 0.01, 0.1],
        "degree": [2, 3],
    },

    "dummy": {  # Dummy Regressor
        # DummyRegressor, almost nothing to tune, also excluded from PyCaret tuning
        "strategy": ["mean", "median"],

    },
}
