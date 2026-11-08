"""Versioned boundary replacement; no boundary-only deletion is permitted."""
from copy import deepcopy

from spatial_creation import CreationRequestError, _parse, _fields, _revision, _text, _geometry
from state_store import RegistryConflictError


def parse_boundary_edit(body):
    value = _parse(body)
    _fields(value, {'entity', 'plot_id', 'geometry', 'geometry_version', 'registry_revision', 'operation'})
    revision = _revision(value)
    entity = value.get('entity')
    if entity not in ('farm', 'plot'):
        raise CreationRequestError('Choose farm or plot.', fields={'entity': 'Choose farm or plot.'})
    plot_id = _text(value.get('plot_id'), 'plot_id') if entity == 'plot' else None
    if entity == 'farm' and 'plot_id' in value:
        raise CreationRequestError('Farm boundaries do not accept plot_id.')
    if 'geometry_version' not in value:
        raise CreationRequestError('Reload the saved boundary before editing.', status=428, code='version_required')
    version = _text(value['geometry_version'], 'geometry_version')
    operation = value.get('operation', 'replace')
    if operation not in ('replace', 'delete'):
        raise CreationRequestError('Unsupported boundary operation.')
    if operation == 'delete':
        raise CreationRequestError('Boundaries are required. Replace the boundary or remove the whole plot in settings.',
                                   status=409, code='boundary_required')
    return dict(entity=entity, plot_id=plot_id, geometry=_geometry(value.get('geometry'), 'geometry'),
                geometry_version=version, registry_revision=revision)


def replace_boundary(store, request, prepare=lambda saved: None):
    """Compare versions and validate/persist the entire layout under one DB lock."""
    with store.transaction():
        saved = store.load_registry()
        if saved['registry_revision'] != request['registry_revision']:
            raise RegistryConflictError('Farm layout changed. Reload before editing again.')
        if saved['farm'] is None:
            raise CreationRequestError('Create the farm before editing boundaries.', status=409, code='setup_required')
        record = saved['farm'] if request['entity'] == 'farm' else next(
            (p for p in saved['plots'] if p['plot_id'] == request['plot_id']), None)
        if record is None:
            raise CreationRequestError('This plot does not belong to the farm.', status=404, code='plot_not_found')
        if record['geometry_version'] != request['geometry_version']:
            raise RegistryConflictError('Boundary changed. Reload before editing again.')
        record['geometry'] = deepcopy(request['geometry'])
        store.save_registry(saved, updated_by='farm_owner')
        prepared = prepare(saved)
    return saved, prepared
