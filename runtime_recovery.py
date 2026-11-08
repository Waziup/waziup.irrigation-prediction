"""Restore per-plot inference state without fitting models or replaying commands.

Only locally generated HMAC-authenticated snapshots are deserialized. Saved
forecasts are display-only; prediction workers must obtain new sensor evidence.
"""
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import re
import sys
from importlib import metadata
from datetime import datetime, timezone

log = logging.getLogger(__name__)
VERSION = 1
FIELDS = ('best_model', 'best_exp', 'data_scaler', 'use_pycaret', 'model_artifacts')


def configuration_digest(plot):
    reader = getattr(plot, 'read_config', None)
    config = reader() if callable(reader) else getattr(plot, 'config', {})
    value = {'plot_id': str(getattr(plot, 'stable_id', None) or plot.id),
             'config': config, 'timezone': getattr(plot, 'timezone', None),
             'csv': getattr(plot, 'load_data_from_csv', False),
             'csv_path': getattr(plot, 'data_from_csv', None) if getattr(plot, 'load_data_from_csv', False) else None}
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def implementation_digest():
    root = Path(__file__).resolve().parent
    names = ('create_model/features.py', 'create_model/nn_training.py',
             'create_model/prediction.py', 'create_model/constants.py',
             'create_model/soil.py', 'sensor_roles.py')
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()



def forecast_implementation_digest():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256(implementation_digest().encode())
    for name in ('crop_model.py', 'phenology_engine.py', 'decision_engine.py',
                 'create_model/orchestration.py', 'recommendation_contract.py'):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()

def dependency_versions():
    versions = {'python': list(sys.version_info[:2])}
    for name in ('pycaret', 'tensorflow', 'scikit-learn', 'numpy', 'pandas', 'cloudpickle'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def model_path(plot):
    identity = str(getattr(plot, 'stable_id', None) or plot.id)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', identity):
        raise ValueError('Invalid plot identity for recovery')
    default = Path(os.getenv('APP_STATE_DB') or 'data/wazifarm.sqlite3').parent / 'runtime_models'
    return Path(os.getenv('IRRIGATION_MODEL_STATE_DIR') or default) / identity / 'inference.snapshot'


def save_model_state(plot):
    """Publish a complete fitted model, experiment and original scaler together."""
    import cloudpickle
    from pipeline_cache import _signing_key
    if getattr(plot, 'best_model', None) is None:
        return False
    payload = {'version': VERSION, 'plot_id': str(getattr(plot, 'stable_id', None) or plot.id),
               'configuration': configuration_digest(plot), 'implementation': implementation_digest(),
               'dependencies': dependency_versions(),
               'saved_at': datetime.now(timezone.utc).isoformat(),
               'state': {field: getattr(plot, field, None) for field in FIELDS}}
    raw = cloudpickle.dumps(payload)
    path = model_path(plot)
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = hmac.new(_signing_key(path, create=True), raw, hashlib.sha256).hexdigest().encode()
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as output:
        os.chmod(temporary, 0o600)
        output.write(signature + b'\n' + raw)
        output.flush()
        os.fsync(output.fileno())
    # A failed serialization/publication leaves the prior generation intact.
    temporary.replace(path)
    plot.runtime_model_saved = True
    return True


def restore_model_state(plot):
    import cloudpickle
    from pipeline_cache import _signing_key
    path = model_path(plot)
    if not path.exists():
        return False, 'model_state_missing'
    try:
        signature, raw = path.read_bytes().split(b'\n', 1)
        key = _signing_key(path, create=False)
        if key is None or not hmac.compare_digest(signature, hmac.new(key, raw, hashlib.sha256).hexdigest().encode()):
            return False, 'model_state_invalid_signature'
        payload = cloudpickle.loads(raw)
        if (payload.get('version') != VERSION or
                payload.get('plot_id') != str(getattr(plot, 'stable_id', None) or plot.id) or
                payload.get('configuration') != configuration_digest(plot) or
                payload.get('implementation') != implementation_digest() or
                payload.get('dependencies') != dependency_versions()):
            return False, 'model_state_incompatible'
        state = payload['state']
        model, scaler = state['best_model'], state['data_scaler']
        if (not callable(getattr(model, 'predict', None)) or
                not callable(getattr(scaler, 'transform', None)) or
                not hasattr(scaler, 'mean_') or
                (state['use_pycaret'] and not callable(getattr(state['best_exp'], 'predict_model', None)))):
            return False, 'model_state_incomplete'
        # Validate the whole payload before changing any runtime field.
        reader = getattr(plot, 'read_config', None)
        config = reader() if callable(reader) else getattr(plot, 'config', {})
        for field in FIELDS:
            setattr(plot, field, state.get(field))
        # Feature construction uses this dictionary, not only plot attributes.
        plot.config = config
        return True, None
    except Exception:
        log.exception('Unable to restore model for plot %s', getattr(plot, 'stable_id', plot.id))
        return False, 'model_state_unreadable'


def recover_plot(plot, max_age_hours):
    """Recover display and model independently; never actuate or start training."""
    import pipeline_cache
    payload, reason = pipeline_cache.load_cycle(plot.id)
    restored = False
    if reason is None:
        if (payload.get('plot_id') != str(getattr(plot, 'stable_id', None) or plot.id) or
                payload.get('configuration_digest') != configuration_digest(plot) or
                payload.get('implementation_digest') != forecast_implementation_digest()):
            reason = 'forecast_snapshot_incompatible'
        else:
            cycle, reason = pipeline_cache.restore_cycle(plot, payload, max_age_hours)
            restored = cycle is not None
    if restored:
        plot.pipeline_result.fallback_reason = 'restart_recovery'
    else:
        pipeline_cache.record_unavailable(plot, reason)
    model_ready, model_reason = restore_model_state(plot)
    plot.training_finished = restored or model_ready
    plot.runtime_recovery_status = 'prediction_resuming' if model_ready else (
        'forecast_restored' if restored else 'unavailable')
    plot.runtime_recovery_reason = model_reason
    return model_ready
