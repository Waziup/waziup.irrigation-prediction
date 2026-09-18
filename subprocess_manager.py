# -*- coding: utf-8 -*-
"""
Subprocess manager for memory-intensive operations (tuning, ensemble)
Runs operations in isolated subprocesses that can be killed to free RAM

@author: felix markwordt
"""

import subprocess
import psutil
import logging
import selectors
import time
import os
import codecs
import io
import signal
from pathlib import Path
#from testifu import run_tuning_pycaret_debug

logger = logging.getLogger(__name__)

Debug_flag = True # DEBUG


class SubprocessManager:
    """Manages subprocess execution for memory-intensive ML operations"""
    
    def __init__(self, max_memory_percent=80, timeout_seconds=3600):
        """
        Initialize subprocess manager
        
        Args:
            max_memory_percent: Kill subprocess if memory exceeds this %
            timeout_seconds: Max execution time before killing
        """
        self.max_memory_percent = max_memory_percent
        self.timeout_seconds = timeout_seconds
        self.process = None
        self.temp_dir = Path('data/subprocess_temp')
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def run_tuning_pycaret_subprocess(
        self,
        tmp_dir,
        best_model_path,
        result_path,
        plot_name,
    ):
        import subprocess
        import sys
        from datetime import datetime
        from pathlib import Path
        import logging

        logger = logging.getLogger(__name__)

        script_path = Path(tmp_dir) / f"tune_pycaret_{plot_name}_{datetime.now().timestamp()}.py"
        result_clean = str(result_path).removesuffix(".pkl")
        previous_result = self._classical_artifact_signature(result_clean)

        script_content = f'''
import os
import sys

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
os.environ["PYCARET_NO_LOGGING"] = "1"

# Add current working directory to path for imports
sys.path.insert(0, os.getcwd())
# Fix for trio + debugpy conflict
sys.excepthook = sys.__excepthook__

import pickle
import traceback
import pandas as pd

from pycaret.regression import load_model, save_model, setup
from pycaret.internal.pipeline import Pipeline

from create_model import tune_one_model

print('[SUBPROCESS] Started tuning:', {plot_name!r})

try:
    tmp_dir = {str(tmp_dir)!r}

    # Load data
    data = pd.read_csv(tmp_dir + "/data.csv")
    data = data.reset_index(drop=True)

    # Load config
    with open(tmp_dir + "/config.pkl", "rb") as f:
        config = pickle.load(f)

    print("[SUBPROCESS] Running setup()")

    exp = setup(
        data=data,
        target=config["target"],
        session_id=config["session_id"],
        ignore_features=config["ignore_features"],
        train_size=config["train_size"],
        normalize=config.get("normalize", False),
        verbose=False,
        log_experiment=False,
        html=False    
    )

    # Load model
    model_path = {str(best_model_path)!r}.removesuffix(".pkl")
    print(f"[SUBPROCESS] Loading model from {{model_path}}")

    best_model = load_model(model_path)

    if isinstance(best_model, list):
        best_model = best_model[0]

    if isinstance(best_model, Pipeline):
        best_model = best_model.steps[-1][1]

    print("[SUBPROCESS] Tuning model...")
    tuned_model = tune_one_model(exp, best_model)

    # Save result
    result_path = {result_clean!r}
    print(f"[SUBPROCESS] Saving model to {{result_path}}")

    save_model(tuned_model, result_path)

    print("[SUCCESS] Tuning completed")
    exit(0)

except Exception as e:
    import traceback
    print("[ERROR FULL TRACE]")
    traceback.print_exc()
    print("ERROR TYPE:", type(e))
    print("ERROR:", repr(e))
    sys.exit(1)
        '''

        try:
            # Write script
            with open(script_path, "w") as f:
                f.write(script_content)

            logger.info(f"Starting subprocess tuning for {plot_name}")

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                close_fds=True
            )

            stdout, stderr = self._run_process(process, plot_name)

            print("----- SUBPROCESS STDOUT -----")
            print(stdout)
            print("----- SUBPROCESS STDERR -----")
            print(stderr)

            if process.returncode == 0 and not getattr(process, '_irrigation_timed_out', False):
                if self._classical_result_is_new(result_clean, previous_result):
                    return result_clean
                else:
                    logger.error("Model file missing after subprocess")
                    return None
            else:
                logger.error(f"Subprocess failed: {stderr}")
                return None

        finally:
            # Delete script
            if script_path.exists():
                script_path.unlink()
    

    def run_ensemble_pycaret_subprocess(
        self,
        tmp_dir,
        model_paths,
        result_path,
        plot_name,
    ):
        import subprocess
        import sys
        from datetime import datetime
        from pathlib import Path
        import logging
        import json

        logger = logging.getLogger(__name__)

        script_path = Path(tmp_dir) / f"ensemble_pycaret_{plot_name}_{datetime.now().timestamp()}.py"
        result_clean = str(result_path).removesuffix(".pkl")
        previous_result = self._classical_artifact_signature(result_clean)

        # pass model paths as json, read them later in subprocess
        model_paths_json = json.dumps(model_paths)

        script_content = f'''
import os
import sys

# ENV FIXES
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
os.environ["PYCARET_NO_LOGGING"] = "1"

# Add current working directory to path for imports
sys.path.insert(0, os.getcwd())
# Fix for trio + debugpy conflict
sys.excepthook = sys.__excepthook__

import pickle
import traceback
import pandas as pd
import json

from pycaret.regression import load_model, save_model, setup

from create_model import create_and_compare_ensemble

print('[SUBPROCESS] Started ENSEMBLE:', {plot_name!r})

try:
    tmp_dir = {str(tmp_dir)!r}

    # Load data
    data = pd.read_csv(tmp_dir + "/data.csv")
    data = data.reset_index(drop=True)

    model_paths = json.loads({model_paths_json!r})

    # Load config
    with open(tmp_dir + "/config.pkl", "rb") as f:
        config = pickle.load(f)

    print("[SUBPROCESS] Running setup()")

    exp = setup(
        data=data,
        target=config["target"],
        session_id=config["session_id"],
        ignore_features=config["ignore_features"],
        train_size=config["train_size"],
        normalize=config.get("normalize", False),
        verbose=False,
        log_experiment=False,
        html=False
    )

    # Load all models
    loaded_models = []
    for path in model_paths:
        clean_path = path.removesuffix(".pkl")
        print(f"[SUBPROCESS] Loading model: {{clean_path}}")
        m = load_model(clean_path)
        loaded_models.append(m)

    print(f"[SUBPROCESS] Loaded {{len(loaded_models)}} models")

    # Run ensemble logic
    best_model = create_and_compare_ensemble(
        {plot_name!r},
        exp,
        loaded_models
    )

    # Save result
    result_path = {result_clean!r}
    print(f"[SUBPROCESS] Saving ensemble model to {{result_path}}")

    save_model(best_model, result_path)

    print("[SUCCESS] Ensemble completed")
    exit(0)

except Exception as e:
    import traceback
    print("[ERROR FULL TRACE]")
    traceback.print_exc()
    print("ERROR TYPE:", type(e))
    print("ERROR:", repr(e))
    sys.exit(1)
'''

        try:
            # Write script
            with open(script_path, "w") as f:
                f.write(script_content)

            logger.info(f"Starting subprocess ENSEMBLE for {plot_name}")

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                close_fds=True
            )

            stdout, stderr = self._run_process(process, plot_name)

            print("----- SUBPROCESS STDOUT -----")
            print(stdout)
            print("----- SUBPROCESS STDERR -----")
            print(stderr)

            if process.returncode == 0 and not getattr(process, '_irrigation_timed_out', False):
                if self._classical_result_is_new(result_clean, previous_result):
                    return result_clean
                else:
                    logger.error("Ensemble model file missing after subprocess")
                    return None
            else:
                logger.error(f"Subprocess ENSEMBLE failed: {stderr}")
                return None

        finally:
            if script_path.exists():
                script_path.unlink()

    
    def run_tuning_and_ensemble_pycaret_subprocess(
        self,
        tmp_dir,
        model_paths,
        result_path,
        plot_name,
    ):
        import subprocess
        import sys
        from datetime import datetime
        from pathlib import Path
        import logging
        import json

        logger = logging.getLogger(__name__)

        script_path = Path(tmp_dir) / f"ensemble_pycaret_{plot_name}_{datetime.now().timestamp()}.py"
        result_clean = str(result_path).removesuffix(".pkl")
        previous_result = self._classical_artifact_signature(result_clean)

        # pass model paths as json, read them later in subprocess
        model_paths_json = json.dumps(model_paths)

        script_content = f'''
import os
import sys

# ENV FIXES - these are needed to prevent debugpy conflicts and reduce overhead in subprocess
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
os.environ["PYCARET_NO_LOGGING"] = "1"
os.environ["PYDEVD_USE_CYTHON"] = "NO"
os.environ["PYDEVD_LOAD_VALUES_ASYNC"] = "0"
os.environ["PYDEVD_UNBLOCK_THREADS_TIMEOUT"] = "0"
os.environ["PYDEVD_THREAD_DUMP_ON_WARN"] = "0"
os.environ["PYDEVD_DISABLE_ATTACH"] = "1"


# Add current working directory to path for imports
sys.path.insert(0, os.getcwd())
# Fix for trio + debugpy conflict
sys.excepthook = sys.__excepthook__
# Disable tracing for subprocess to reduce overhead and avoid debugpy conflicts
sys.settrace(None)  

import pickle
import traceback
import pandas as pd
import json

from pycaret.regression import load_model, save_model, setup
from pycaret.internal.pipeline import Pipeline

from create_model import create_and_compare_ensemble, tune_models, save_models

print('[SUBPROCESS] Started TUNING and ENSEMBLE in one subprocess for Plot:', {plot_name!r})

try:
    tmp_dir = {str(tmp_dir)!r}

    # Load data
    data = pd.read_csv(tmp_dir + "/data.csv")
    data = data.reset_index(drop=True)

    model_paths = json.loads({model_paths_json!r})

    # Load config
    with open(tmp_dir + "/config.pkl", "rb") as f:
        config = pickle.load(f)

    print("[SUBPROCESS] Running setup()")

    exp = setup(
        data=data,
        target=config["target"],
        session_id=config["session_id"],
        ignore_features=config["ignore_features"],
        train_size=config["train_size"],
        normalize=config.get("normalize", False),
        verbose=False,
        log_experiment=False,
        html=False
    )

    # Load all models
    loaded_models = []
    for path in model_paths:
        clean_path = path.removesuffix(".pkl")
        print(f"[SUBPROCESS] Loading model: {{clean_path}}")
        m = load_model(clean_path)
        if isinstance(m, Pipeline):
            m = m.steps[-1][1]
        loaded_models.append(m)

    print(f"[SUBPROCESS] Loaded {{len(loaded_models)}} models")

    if isinstance(loaded_models, Pipeline):
        loaded_models = loaded_models.steps[-1][1]

    # Run tuning in subprocess
    tuned_best_models = tune_models(exp, loaded_models) 

    for m in tuned_best_models:
        if isinstance(m, Pipeline):
            m = m.steps[-1][1]

    # Save best tuned pycaret model
    model_names = save_models({plot_name!r}, exp, tuned_best_models, {('models/' + plot_name + '/tuned_models/pycaret/best_soil_tension_prediction_')!r})

    # Run ensemble logic
    ensemble_best_model = create_and_compare_ensemble(
        {plot_name!r},
        exp,
        tuned_best_models
    )

    # Save result
    result_path = {result_clean!r}
    print(f"[SUBPROCESS] Saving ensemble model to {{result_path}}")

    save_model(ensemble_best_model, result_path)

    print("[SUCCESS] Ensemble completed")
    exit(0)

except Exception as e:
    import traceback
    print("[ERROR FULL TRACE]")
    traceback.print_exc()
    print("ERROR TYPE:", type(e))
    print("ERROR:", repr(e))
    sys.exit(1)
'''

        try:
            # Write script
            with open(script_path, "w") as f:
                f.write(script_content)

            logger.info(f"Starting subprocess ENSEMBLE for {plot_name}")

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                close_fds=True
            )

            stdout, stderr = self._run_process(process, plot_name)

            print("----- SUBPROCESS STDOUT -----")
            print(stdout)
            print("----- SUBPROCESS STDERR -----")
            print(stderr)

            if process.returncode == 0 and not getattr(process, '_irrigation_timed_out', False):
                if self._classical_result_is_new(result_clean, previous_result):
                    return result_clean
                else:
                    logger.error("Ensemble model file missing after subprocess")
                    return None
            else:
                logger.error(f"Subprocess ENSEMBLE failed: {stderr}")
                return None

        finally:
            if script_path.exists():
                script_path.unlink()


    @staticmethod
    def _classical_artifact_signature(prefix):
        """Inspect the exact PyCaret output file without deleting prior models."""
        path = Path(str(prefix) + '.pkl')
        try:
            if path.is_symlink() or not path.is_file():
                return None
            info = path.stat()
            if info.st_size == 0:
                return None
            return (info.st_dev, info.st_ino, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns)
        except OSError:
            return None

    @classmethod
    def _classical_result_is_new(cls, prefix, previous):
        current = cls._classical_artifact_signature(prefix)
        return current is not None and current != previous

    @staticmethod
    def _read_nn_result(status_path, count=1):
        """Accept a completed result only when all reported files exist."""
        try:
            paths = Path(status_path).read_text().strip().split('|')
            if len(paths) != count or any(not path.strip() for path in paths):
                raise ValueError('Malformed NN subprocess result')
            paths = [path.strip() for path in paths]
            if any(not Path(path).is_file() or Path(path).stat().st_size == 0 for path in paths):
                raise ValueError('NN subprocess result references missing or empty artifacts')
            return tuple(paths) if count > 1 else paths[0]
        except (OSError, ValueError) as exc:
            logger.warning('Invalid NN subprocess result %s: %s', status_path, exc)
            return None

    def run_tune_single_model_subprocess(self, tmp_dir, model_config, plot_name):
        import subprocess, sys, json
        from pathlib import Path
        from datetime import datetime

        script_path = Path(tmp_dir) / f"nn_tune_single_{datetime.now().timestamp()}.py"
        status_path = Path(tmp_dir) / 'result_single.txt'
        # Calls within a handoff directory are sequential. Discard any previous
        # candidate's status before starting this child, then consume its status.
        status_path.unlink(missing_ok=True)
        model_config_json = json.dumps(model_config)

        script_content = f'''
import os, sys, json, gc
import numpy as np
import tensorflow as tf

# Reduce TF overhead
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Add current working directory to path for imports
sys.path.insert(0, os.getcwd())

from create_model import tune_model_nn, Model_functions, save_models_nn
from keras_tuner.engine.hyperparameters import HyperParameters

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tmp_dir = {str(tmp_dir)!r}

X_train = np.load(tmp_dir + "/X_train.npy")
y_train = np.load(tmp_dir + "/y_train.npy")
X_val   = np.load(tmp_dir + "/X_val.npy")
y_val   = np.load(tmp_dir + "/y_val.npy")

cfg = json.loads({model_config_json!r})

# Build model
hp = HyperParameters()
if cfg["hp_values"]:
    for k, v in cfg["hp_values"].items():
        hp.values[k] = v

model = Model_functions[cfg["model_name"]](hp, tuple(cfg["shape"]))

if cfg.get("weights_path") and os.path.exists(cfg["weights_path"]):
    model.load_weights(cfg["weights_path"])

# Tune
tuned_model, tuned_hp = tune_model_nn(X_train, y_train, X_val, y_val, model)

# Save
paths = save_models_nn({plot_name!r}, tuned_model, tmp_dir + "/", [tuned_hp])

model_path = paths[0]

# Save HP JSON next to model
hp_path = model_path.replace(".keras", ".json")

with open(hp_path, "w") as f:
    json.dump(tuned_hp.values, f)

# Return BOTH paths
with open(tmp_dir + "/result_single.txt", "w") as f:
    f.write(model_path + "|" + hp_path)

tf.keras.backend.clear_session()
gc.collect()
'''

        try:
            with open(script_path, "w") as f:
                f.write(script_content)

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                close_fds=True
            )
            if Debug_flag:
                stdout, stderr = self._run_process_stream(process, plot_name)
            else:
                stdout, stderr = self._run_process(process, plot_name)

            print(stdout)
            print(stderr)

            if process.returncode != 0 or getattr(process, '_irrigation_timed_out', False):
                print(stderr)
                return None
            return self._read_nn_result(status_path, count=2)
        finally:
            if script_path.exists():
                script_path.unlink()
            status_path.unlink(missing_ok=True)
        
    def run_ensemble_nn_subprocess(self, tmp_dir, model_paths, hp_paths, plot_name):
        import subprocess, sys, json
        from pathlib import Path
        from datetime import datetime

        script_path = Path(tmp_dir) / f"nn_ensemble_{datetime.now().timestamp()}.py"
        status_path = Path(tmp_dir) / 'ensemble_result.txt'
        status_path.unlink(missing_ok=True)
        model_paths_json = json.dumps(model_paths)
        hp_paths_json = json.dumps(hp_paths)

        script_content = f'''
import os, sys, json, gc
import numpy as np
import tensorflow as tf

# Add current working directory to path for imports
sys.path.insert(0, os.getcwd())

from create_model import compare_nn_ensembles, load_models_nn, save_models_nn

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tmp_dir = {str(tmp_dir)!r}

X_train = np.load(tmp_dir + "/X_train.npy")
y_train = np.load(tmp_dir + "/y_train.npy")
X_val   = np.load(tmp_dir + "/X_val.npy")
y_val   = np.load(tmp_dir + "/y_val.npy")

model_paths = json.loads({model_paths_json!r})
hp_paths = json.loads({hp_paths_json!r})

models = []
hps = []

for mp, hp in zip(model_paths, hp_paths):
    print(f"[SUBPROCESS ENSEMBLE] Model: {{mp}}")
    print(f"[SUBPROCESS ENSEMBLE] HP: {{hp}}")

    loaded = load_models_nn(mp)

    # Case 1: returns tuple (model, something)
    if isinstance(loaded, tuple):
        loaded = loaded[0]

    # Case 2: returns list of models
    if isinstance(loaded, list):
        models.extend(loaded)
    else:
        models.append(loaded)

    with open(hp, "r") as f:
        hps.append(json.load(f))

results = compare_nn_ensembles(
    models,
    hps,
    X_train,
    y_train,
    X_val,
    y_val
)

best_model = results["best_predictor"]

# Save best
# best_model.save(tmp_dir + "/ensemble.keras")
save_path = save_models_nn({plot_name!r}, best_model, tmp_dir + "/", None)

with open(tmp_dir + "/ensemble_result.txt", "w") as f:
    f.write(save_path[0])

tf.keras.backend.clear_session()
gc.collect()
'''

        try:
            with open(script_path, "w") as f:
                f.write(script_content)

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                close_fds=True
            )
            if Debug_flag:
                stdout, stderr = self._run_process_stream(process, plot_name)
            else:
                stdout, stderr = self._run_process(process, plot_name)

            print(stdout)
            print(stderr)

            if process.returncode != 0 or getattr(process, '_irrigation_timed_out', False):
                print(stderr)
                return None
            return self._read_nn_result(status_path)
        finally:
            if script_path.exists():
                script_path.unlink()
            status_path.unlink(missing_ok=True)
    

    def run_tuning_and_ensemble_nn_subprocess(
        self,
        tmp_dir,
        model_configs,
        result_path,
        plot_name,
    ):
        import subprocess
        import sys
        from datetime import datetime
        from pathlib import Path
        import json
        import logging

        logger = logging.getLogger(__name__)

        script_path = Path(tmp_dir) / f"nn_subprocess_{plot_name}_{datetime.now().timestamp()}.py"
        status_path = Path(tmp_dir) / 'result_path.txt'
        status_path.unlink(missing_ok=True)
        result_clean = str(result_path).replace(".keras", "")

        model_configs_json = json.dumps(model_configs)

        script_content = f'''
print('[SUBPROCESS NN] Started tuning and ensemble with Neural Networks for:', {plot_name!r}, flush=True)
import os
import sys

# Reduce TF overhead
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.getcwd())
if sys.gettrace() is None:
    # sys.settrace(None)
    pass  # Not debugging, no need to disable tracing
sys.stdout.reconfigure(line_buffering=True)

import json
import pandas as pd
import numpy as np
import tensorflow as tf
import gc

from create_model import tune_model_nn, compare_nn_ensembles, Model_functions, save_models_nn
from keras_tuner.engine.hyperparameters import HyperParameters

print('[SUBPROCESS NN] Started for:', {plot_name!r})

try:
    tmp_dir = {str(tmp_dir)!r}

    X_train = np.load(tmp_dir + "/X_train.npy")
    y_train = np.load(tmp_dir + "/y_train.npy")
    X_val   = np.load(tmp_dir + "/X_val.npy")
    y_val   = np.load(tmp_dir + "/y_val.npy")

    model_configs = json.loads({model_configs_json!r})

    print("[SUBPROCESS NN] Rebuilding models")

    models = []

    for cfg in model_configs:
        hp = HyperParameters()

        # Set hyperparameters from config
        if cfg["hp_values"]:
            for k, v in cfg["hp_values"].items():
                hp.values[k] = v

        # Build model
        model = Model_functions[cfg["model_name"]](hp, tuple(cfg["shape"]))
        
        # Load weights
        if cfg.get("weights_path") and os.path.exists(cfg["weights_path"]):
            model.load_weights(cfg["weights_path"])
        
        model.model_name = cfg["model_name"]
        model.shape = tuple(cfg["shape"])

        models.append(model)

    print("[SUBPROCESS NN] Tuning models")

    tuned_models = []
    tuned_hps = []

    for m in models:
        tuned, hp = tune_model_nn(X_train, y_train, X_val, y_val, m)
        tuned_models.append(tuned)
        tuned_hps.append(hp)

    tf.keras.backend.clear_session()
    gc.collect()

    save_models_nn({plot_name!r}, tuned_models, {('models/' + plot_name + '/tuned_models/nn/soil_tension_prediction_')!r}, tuned_hps)

    print("[SUBPROCESS NN] Running ensemble")

    results = compare_nn_ensembles(
        tuned_models,
        tuned_hps,
        X_train,
        y_train,
        X_val,
        y_val
    )

    # Only save the best model from ensemble
    best_model = results["best_predictor"]

    # Save best model to tmp dir -> DEBUG: skipping ensemble for now to speed up, just return best tuned model
    best_model_paths = save_models_nn({plot_name!r}, best_model, tmp_dir + "/", None)

    with open(tmp_dir + "/result_path.txt", "w") as f:
        f.write(best_model_paths[0])

    # Cleanup
    tf.keras.backend.clear_session()
    gc.collect()

    print("[SUCCESS] Tuning and ensemble with Neural Networks: Done")
    exit(0)

except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

        try:
            with open(script_path, "w") as f:
                f.write(script_content)

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                close_fds=True
            )
            if Debug_flag:
                stdout, stderr = self._run_process_stream(process, plot_name)
            else:
                stdout, stderr = self._run_process(process, plot_name)

            print(stdout)
            print(stderr)

            if process.returncode == 0 and not getattr(process, '_irrigation_timed_out', False):
                return self._read_nn_result(status_path)
            return None

        finally:
            if script_path.exists():
                script_path.unlink()
            status_path.unlink(missing_ok=True)

    def _monitor_process(self, plot_name, check_interval=5):
        """
        Monitor subprocess for timeout and memory usage
        
        Args:
            plot_name: Name for logging
            check_interval: How often to check (seconds)
            
        Returns:
            Tuple of (stdout, stderr)
        """
        import time
        
        start_time = time.time()
        
        try:
            stdout, stderr = self.process.communicate(timeout=self.timeout_seconds)
            elapsed = time.time() - start_time
            logger.info(f"[{plot_name}] Process completed in {elapsed:.1f}s")
            return stdout, stderr
            
        except subprocess.TimeoutExpired:
            logger.warning(f"[{plot_name}] Subprocess timeout ({self.timeout_seconds}s), killing process")
            self.process.kill()
            stdout, stderr = self.process.communicate()
            return stdout, stderr
        
        except Exception as e:
            logger.error(f"[{plot_name}] Error monitoring process: {e}")
            if self.process.poll() is None:
                self.process.kill()
            return "", str(e)
    
    def _cleanup(self, script_path):
        """Clean up temporary script file"""
        try:
            if script_path.exists():
                script_path.unlink()
        except Exception as e:
            logger.warning(f"Could not delete temp script {script_path}: {e}")
    
    def cleanup_old_temps(self, max_age_hours=24):
        """Clean up old temporary files"""
        import time
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        for f in self.temp_dir.glob("*.py"):
            if f.stat().st_mtime < cutoff_time:
                try:
                    f.unlink()
                    logger.debug(f"Cleaned up {f.name}")
                except Exception as e:
                    logger.warning(f"Could not delete {f}: {e}")

    def _run_process(self, process, plot_name):
        # A parent can exit zero while workers keep its pipes open. Preserve
        # deadline failure separately from the OS exit code for result callers.
        process._irrigation_timed_out = False
        self._remember_process_group(process)
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
            process.wait()

        except subprocess.TimeoutExpired:
            process._irrigation_timed_out = True
            logger.warning(f"{plot_name}: timeout → killing process tree")
            self._kill_process_tree(process)
            stdout, stderr = process.communicate()
            return None, None

        finally:
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()

            if process.poll() is None:
                self._kill_process_tree(process)

            import gc
            gc.collect()

        return stdout, stderr
    
    def _run_process_stream(self, process, plot_name):
        """Stream both pipes while enforcing the manager's real timeout."""
        process._irrigation_timed_out = False
        self._remember_process_group(process)
        stdout_lines = []
        stderr_lines = []
        selector = selectors.DefaultSelector()
        deadline = time.monotonic() + self.timeout_seconds
        decoders = {}
        try:
            for pipe, label in ((process.stdout, 'STDOUT'), (process.stderr, 'STDERR')):
                if pipe:
                    os.set_blocking(pipe.fileno(), False)
                    decoder = codecs.getincrementaldecoder(getattr(pipe, 'encoding', None) or 'utf-8')(errors='replace')
                    if getattr(process, 'text_mode', False):
                        decoder = io.IncrementalNewlineDecoder(decoder, translate=True)
                    decoders[label] = decoder
                    selector.register(pipe, selectors.EVENT_READ, label)
            # EOF on both pipes does not imply that the child has exited.
            while selector.get_map() or process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process._irrigation_timed_out = True
                    logger.warning(
                        "%s: streaming subprocess timeout after %ss; killing process tree",
                        plot_name,
                        self.timeout_seconds,
                    )
                    self._kill_process_tree(process)
                    break

                for key, _ in selector.select(timeout=min(0.05, remaining)):
                    # Read available bytes, never wait for a newline. Incremental
                    # decoding preserves characters split across pipe reads.
                    try:
                        chunk = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    line = decoders[key.data].decode(chunk, final=not chunk)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        decoders.pop(key.data)
                    if line:
                        print(f"[{plot_name} {key.data}] {line}", end="")
                        (stdout_lines if key.data == 'STDOUT' else stderr_lines).append(line)
        finally:
            selector.close()
            if process.poll() is None:
                self._kill_process_tree(process)
            process.wait()
            for pipe in (process.stdout, process.stderr):
                if pipe:
                    pipe.close()

        for label, decoder in decoders.items():
            tail = decoder.decode(b'', final=True)
            (stdout_lines if label == 'STDOUT' else stderr_lines).append(tail)

        return "".join(stdout_lines), "".join(stderr_lines)

    def _remember_process_group(self, process):
        """Remember an isolated child's group before poll() can reap its leader."""
        if os.name != 'posix':
            return
        try:
            group = os.getpgid(process.pid)
            if group == process.pid and group != os.getpgrp():
                process._irrigation_process_group = group
        except ProcessLookupError:
            pass

    def _kill_process_tree(self, process):
        """Kill owned workers, including those whose group leader has exited."""
        group = getattr(process, '_irrigation_process_group', None)
        if os.name == 'posix' and group == process.pid and group != os.getpgrp():
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                logger.warning('Failed to kill subprocess group %s: %s', group, exc)
        # Also handle callers that do not launch an isolated process group.
        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)

            logger.warning(f"Killing {len(children)} child processes")

            for child in children:
                try:
                    child.kill()
                except Exception as e:
                    logger.warning(f"Failed to kill child {child.pid}: {e}")

            parent.kill()

        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            logger.warning(f"Failed to kill process tree: {e}")


def run_tuning_with_subprocess(temp_dir, best_model_path, plot_name):
    """
    Convenience function to run PyCaret tuning in subprocess
    
    Returns tuned model path after loading and saving from disk
    """

    try:   
        manager = SubprocessManager(max_memory_percent=85, timeout_seconds=3600)
        
        # DEBUG: run tuning inline without subprocess to get better error messages
        run_as_subprocess = True  # Set to False to run inline for debugging      
        
        if run_as_subprocess:
            # Run tuning in subprocess
            result_path = manager.run_tuning_pycaret_subprocess(
                str(temp_dir), 
                str(best_model_path), 
                str(temp_dir + "/" + "result_model.pkl"),
                plot_name
            )
        else:
            # Run tuning inline for debugging
            result_path = run_tuning_pycaret_debug(
                str(temp_dir), 
                str(best_model_path), 
                str(temp_dir + "/" + "result_model.pkl"),
                plot_name
            )

        if result_path:
            return result_path
        # throw error if tuning failed and no result returned
        raise RuntimeError(f"Tuning failed for {plot_name}, no result returned")
    except Exception as e:
        logger.error(f"Error during tuning for {plot_name}: {e}")
        raise


def run_ensemble_with_subprocess(self,temp_dir, best_model_path, hp_paths, plot_name):
    """
    Convenience function to run PyCaret ensemble creation in subprocess
    
    Returns ensemble model after loading from disk
    """
    
    manager = SubprocessManager(max_memory_percent=85, timeout_seconds=3600)

    try:        
        # Run ensemble in subprocess
        result_path = manager.run_ensemble_pycaret_subprocess(
            str(temp_dir), 
            best_model_path,
            str(temp_dir + "/" + "result_model.pkl"),
            plot_name
        )
        
        if result_path:
            return result_path
        # throw error if tuning failed and no result returned
        raise RuntimeError(f"Tuning failed for {plot_name}, no result returned")
    except Exception as e:
        logger.error(f"Error during tuning for {plot_name}: {e}")
        raise

def run_tuning_and_ensemble_with_subprocess(temp_dir, best_model_path, plot_name):
    """
    Convenience function to run PyCaret ensemble creation in subprocess
    
    Returns ensemble model after loading from disk
    """
    
    manager = SubprocessManager(max_memory_percent=85, timeout_seconds=3600)

    try:        
        # Run ensemble in subprocess
        result_path = manager.run_tuning_and_ensemble_pycaret_subprocess(
            str(temp_dir), 
            best_model_path,
            str(temp_dir + "/" + "result_model.pkl"),
            plot_name
        )
        
        if result_path:
            return result_path
        # throw error if tuning failed and no result returned
        raise RuntimeError(f"Tuning failed for {plot_name}, no result returned")
    except Exception as e:
        logger.error(f"Error during tuning for {plot_name}: {e}")
        raise

def run_full_nn_pipeline(self, tmp_dir, model_configs, plot_name):
    """
    Orchestrates:
    1. Tune each model in isolated subprocess
    2. Run ensemble in separate subprocess
    """

    tuned_model_paths = []
    tuned_hp_paths = []

    print(f"[ORCHESTRATOR] Tuning {len(model_configs)} models...")

    # ---- STEP 1: Tune each model separately ----
    for i, cfg in enumerate(model_configs):
        print(f"[ORCHESTRATOR] Tuning model {i+1}/{len(model_configs)}")

        path = self.run_tune_single_model_subprocess(
            tmp_dir,
            cfg,
            plot_name
        )

        if path is None:
            print(f"[ORCHESTRATOR] Model {i+1} failed → skipping")
            continue


        model_path, hp_path = path

        tuned_model_paths.append(model_path)
        tuned_hp_paths.append(hp_path)

    if not tuned_model_paths:
        raise RuntimeError("All model tunings failed")

    print(f"[ORCHESTRATOR] {len(tuned_model_paths)} models tuned successfully")

    # ---- STEP 2: Ensemble ----
    print("[ORCHESTRATOR] Running ensemble...")

    ensemble_path = self.run_ensemble_nn_subprocess(
        tmp_dir,
        tuned_model_paths,
        tuned_hp_paths,
        plot_name
    )

    # fallback is needed because ensemble subprocess might fail due to OOM or other issues, in that case we return the best tuned model
    if ensemble_path is None:
        print("[ORCHESTRATOR] Ensemble failed, returning best tuned model")

        # In case of ensemble failure, return the best tuned model (first one in the list)
        ensemble_path = tuned_model_paths[0]

    return ensemble_path

def run_tuning_and_ensemble_nn_with_subprocess(temp_dir, model_configs, plot_name):

    run_individual_subprocess = True  # Set to True to run tuning of individual models in separate subprocesses, False to run tuning and ensemble in one subprocess

    manager = SubprocessManager(max_memory_percent=85, timeout_seconds=3600)

    if run_individual_subprocess:
        result_path = run_full_nn_pipeline(
            manager,
            str(temp_dir),
            model_configs,
            plot_name
    )
    else:
        # Run tuning and ensemble in one subprocess
        result_path = manager.run_tuning_and_ensemble_nn_subprocess(
            str(temp_dir),
            model_configs,
            str(temp_dir) + "/result_path.txt",
            plot_name
    )

    if result_path:
        return result_path

    raise RuntimeError(f"NN subprocess failed for {plot_name}")
