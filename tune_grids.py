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

    "lar": {  # Least Angle Regression
        "fit_intercept": [True, False],
        "normalize": [True, False],  # NOTE: "normalize" is deprecated in newer sklearn
    },

    "llar": {  # Lasso Least Angle Regression
        "fit_intercept": [True, False],
        "positive": [False, True],
    },

    "omp": {  # Orthogonal Matching Pursuit
        # I'm not fully sure which params PyCaret exposes here.
        # Reasonable guesses based on sklearn:
        "n_nonzero_coefs": [None, 5, 10, 20],
        "fit_intercept": [True, False],
        "normalize": [True, False],  # (deprecated in newer sklearn)
    },

    "br": {  # Bayesian Ridge
        # I'm not fully sure which of these PyCaret passes through,
        # but they are standard sklearn params:
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
        # RANSAC wraps a base estimator (often LinearRegression),
        # and tuning it is tricky. These are tentative:
        "min_samples": [0.1, 0.25, 0.5],
        "residual_threshold": [1.0, 5.0, 10.0],
        "max_trials": [100, 500],
        # I'm not fully sure that PyCaret exposes all of these for tuning.
    },

    "tr": {  # TheilSen Regressor
        # Very expensive for large data; tune carefully.
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
    "mlp": {  # MLP Regressor
        # MLPs are sensitive; this is a very small grid on purpose.
        "hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-4, 1e-3, 1e-2],
        "learning_rate": ["constant", "adaptive"],
        "max_iter": [200, 500],
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

    "lightgbm": {  # LGBMRegressor
        "n_estimators": [200, 500],
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_child_samples": [20, 40, 60],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "reg_lambda": [0, 1, 5],
    },

    "catboost": {  # CatBoostRegressor
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "iterations": [300, 600, 1000],
        "l2_leaf_reg": [1, 3, 5],
        # NOTE: CatBoost has many more params (border_count, bagging, etc.);
        # I’m keeping this short-ish.
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
        # DummyRegressor has almost nothing to tune:
        "strategy": ["mean", "median"],
        # (not sure PyCaret exposes this for tuning, but it's harmless to include)
    },
}
