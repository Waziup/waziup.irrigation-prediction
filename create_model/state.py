"""Mutable runtime state shared across the create_model package and workers.

Runtime flags live here so all package modules and workers observe the same values.
Model serialization is owned by ``model_operation``; callers must not manipulate
``Currently_active`` directly.
"""

from contextlib import contextmanager
import threading

# Prevents concurrent training/prediction of multiple plots (see contract above)
Currently_active = False
# Shared lock: only one training or prediction operation may use the ML stack.
_model_lock = threading.Lock()


@contextmanager
def model_operation():
    """Serialize expensive model work and always release its runtime flag."""
    global Currently_active
    _model_lock.acquire()
    Currently_active = True
    try:
        yield
    finally:
        # Always unblock later cycles, including after model failures.
        Currently_active = False
        _model_lock.release()

## DEBUG -> is overwritten by .env
# to skip data preprocessing and training, load data from file
SkipDataPreprocessing = False       # if true, load dataset from static file
SkipTraining = False                # if true, load predictions from static file
# Load variables of training from file, that had been saved from former training/predictions to debug actuation part: DEBUG
Perform_training = True             # kind of redundant, but automatically saves and loads former results of predictions
Use_subprocess = True               # if true, parts of training is performed in subprocess, to prevent memory leaks and to ensure that resources are released after training

# Verbose logging -> in production this should be false to reduce log size
Verbose_logging = True

# Pycaret regression setup config (populated by pycaret_models at training time,
# read by the subprocess init functions)
Config = {}
