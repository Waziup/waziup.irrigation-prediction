"""create_model package: irrigation-prediction training & prediction pipeline.

Refactored from the original single-file create_model.py. The public surface
(every name importable as `from create_model import X` before the refactor)
is re-exported here, so main.py, the worker threads and the generated
subprocess scripts keep working unchanged.

Mutable runtime flags (Currently_active & friends) live in create_model.state;
reads of them via `create_model.<flag>` still work (module __getattr__ below),
but WRITES must target create_model.state.<flag> - see state.py docstring.
"""

from . import state
from .constants import *

from .runtime import (
    HardCleanupCallback,
    MemoryLimitCallback,
    MemoryLimitReachedError,
    TimeLimitCallback,
    custom_exception_hook,
    free_memory,
)
from .cleaning import (
    convert_cols,
    fill_gaps,
    remove_large_gaps,
    resample,
)
from .weather import (
    get_historical_weather_api,
    get_weather_forecast_api,
    only_get_historical_weather_api,
)
from .soil import (
    add_volumetric_col_to_df,
    calc_volumetric_water_content_single_value,
    soil_tension_to_volumetric_water_content_spline,
)
from .features import (
    Weather_derived_feature_cols,
    add_pump_state,
    add_weather_derived_features,
    create_features,
    ensure_json_file,
    include_irrigation_amount,
    prepare_data,
    split_by_ratio,
)
from .nn_architectures import (
    Model_functions,
    adapt_X_for_model,
    build_optimizer,
    create_cnn_model,
    create_gru_model,
    create_lstm_model,
    create_nn_model,
    create_rnn_model,
    fresh_optimizer_from_model_or_hp,
    hp_get,
    model_builder_with_shape,
    prepare_lstm_data,
    safe_model_name,
)
from .evaluation import (
    eval_approach_mix,
    evaluate_against_testset,
    evaluate_against_testset_nn,
    evaluate_against_validation,
    evaluate_against_validation_nn,
    evaluate_results_and_choose_best,
    evaluate_results_and_choose_top_n,
    evaluate_target_variable,
    get_r2_manual,
)
from .nn_ensemble import (
    EnsemblePredictor,
    compare_nn_ensembles,
)
from .nn_training import (
    init_nn_subprocess_tuning_and_ensemble,
    load_models_nn,
    plot_history_png,
    prepare_data_for_cnn2,
    prepare_future_values,
    save_history_json,
    save_models_nn,
    save_weights,
    train_best_nn,
    train_nn_models,
    tune_model_nn,
)
from .pycaret_models import (
    create_and_compare_ensemble,
    create_and_compare_model_reg,
    init_pycaret_subprocess_ensemble,
    init_pycaret_subprocess_tuning,
    init_pycaret_subprocess_tuning_and_ensemble,
    load_models,
    log_scores_csv,
    safe_make,
    save_models,
    train_best,
    tune_models,
    tune_one_model,
    unwrap_model,
)
from .prediction import (
    align_with_latest_sensor_values,
    calc_threshold,
    compare_train_predictions_cols,
    create_future_values,
    exponential_weights,
    generate_predictions,
    generate_predictions_nn,
    quadratic_weights,
)
from .orchestration import (
    data_pipeline,
    main,
    predict_with_updated_data,
)
from . import legacy


_STATE_FLAGS = ("Currently_active", "Config", "SkipDataPreprocessing", "SkipTraining",
                "Perform_training", "Use_subprocess", "Verbose_logging")

def __getattr__(name):
    # Backward-compat READ access: create_model.Currently_active etc. delegate to
    # the live state module. (PEP 562; writes cannot be intercepted this way -
    # writers were updated to use create_model.state directly.)
    if name in _STATE_FLAGS:
        return getattr(state, name)
    raise AttributeError(f"module 'create_model' has no attribute {name!r}")
