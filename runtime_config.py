"""
Minimal runtime configuration for training and prediction threads.

Extracted from the legacy create_model.py to eliminate unnecessary imports.
"""

# Whether to run training/prediction cycles (vs. loading from cache)
Perform_training = True

# Backoff time used after thread errors before retrying work.
Resource_wait_time_seconds = 1800

# Fallback global base-model key used for unseen farms.
Global_base_model_key = "global_base"

# Enable periodic per-farm retraining when local farm datasets exist.
# Keep False for pure global-base inference deployments.
Enable_farm_specific_training = False

# Maximum age for cached fallback predictions before they are considered stale.
Max_fallback_prediction_age_hours = 24
