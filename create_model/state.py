"""Mutable runtime state shared across the create_model package AND across threads.

Cross-thread contract: training_thread.py and prediction_thread.py read and WRITE
these flags (most importantly Currently_active, the mutex that prevents concurrent
training/prediction). All access - internal and external - must go through THIS
module (create_model.state.X) so every reader sees every writer's update. Do not
copy these values via `from .state import X` (that freezes the value at import time).
"""

# Prevents concurrent training/prediction of multiple plots (see contract above)
Currently_active = False

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
