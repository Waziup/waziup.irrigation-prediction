"""Journal a configuration archive until its registry removal has committed."""

import hashlib
import json
import logging
import os
from pathlib import Path
import re

log = logging.getLogger(__name__)


def record_removal(config, archive, plot_id):
    journal = config.with_suffix(config.suffix + '.removal.json')
    if journal.exists():
        raise RuntimeError(f'Unreconciled removal journal: {journal}')
    payload = {'version': 1, 'plot_id': plot_id, 'config': config.name,
               'archive': archive.name,
               'sha256': hashlib.sha256(config.read_bytes()).hexdigest()}
    temp = journal.with_suffix(journal.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, journal)
    return journal


def clear_journal(journal):
    if journal is not None:
        try:
            journal.unlink(missing_ok=True)
        except OSError:
            # Keeping a completed journal is safe: startup reconciles it again.
            log.exception('Could not clear removal journal %s', journal)


def recover_removals(config_dir, registry):
    root = Path(config_dir)
    plots = {plot['plot_id']: plot for plot in registry['plots']}
    for journal in sorted(root.glob('current_config_plot*.json.removal.json')):
        if journal.is_symlink():
            raise ValueError(f'Unsafe removal journal: {journal}')
        payload = json.loads(journal.read_text(encoding='utf-8'))
        name = payload.get('config', '') if isinstance(payload, dict) else ''
        archive_name = payload.get('archive', '') if isinstance(payload, dict) else ''
        if (not isinstance(name, str) or not re.fullmatch(r'current_config_plot\d+\.json', name)
                or journal.name != name + '.removal.json'
                or not isinstance(archive_name, str)
                or not re.fullmatch(re.escape(name) + r'\.removed(?:\.\d+)?', archive_name)
                or payload.get('version') != 1
                or not isinstance(payload.get('plot_id'), str)
                or not isinstance(payload.get('sha256'), str)):
            raise ValueError(f'Invalid removal journal: {journal}')
        config, archive = root / name, root / archive_name
        if config.is_symlink() or archive.is_symlink():
            raise ValueError(f'Unsafe removal recovery paths: {journal}')
        plot = plots.get(payload['plot_id'])
        if plot is not None and plot['config_file'] != name:
            raise ValueError(f'Removal journal does not match registry: {journal}')
        if plot is None and any(item['config_file'] == name for item in plots.values()):
            raise ValueError(f'Removal target belongs to a different plot: {journal}')
        expected = payload['sha256']
        for path in (config, archive):
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Removal recovery content changed: {path}')
        if plot is not None:
            if config.exists() and archive.exists():
                raise ValueError(f'Conflicting removal recovery files: {journal}')
            if not config.exists():
                if not archive.exists():
                    raise ValueError(f'Missing removal recovery file: {journal}')
                archive.rename(config)
        elif config.exists() or not archive.exists():
            raise ValueError(f'Incomplete committed removal: {journal}')
        clear_journal(journal)
