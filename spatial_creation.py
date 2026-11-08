"""JSON request validation for boundary-first creation; no HTTP or network I/O."""
import json
import math
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from crops import CROP_PARAMS
from geometry_validation import BoundaryValidationError, validate_polygon


class CreationRequestError(ValueError):
    def __init__(self, message, *, status=400, code='invalid_request', fields=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.fields = fields or {}


def _error(field, message):
    raise CreationRequestError(message, fields={field: message})


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CreationRequestError('Duplicate JSON properties are not allowed.')
        result[key] = value
    return result


def _constant(value):
    raise CreationRequestError('Non-finite JSON numbers are not allowed.')


def _parse(body):
    try:
        value = json.loads(body, object_pairs_hook=_object, parse_constant=_constant)
    except CreationRequestError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise CreationRequestError('Send a valid JSON object; form submissions are not supported.') from exc
    if not isinstance(value, dict):
        raise CreationRequestError('Send a JSON object.')
    return value


def _fields(value, allowed, prefix=''):
    unknown = set(value) - allowed
    if unknown:
        key = sorted(unknown)[0]
        _error(prefix + key, 'Unsupported field. Areas are calculated from boundaries; IDs are assigned by the server.')


def _text(value, field, limit=160):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        _error(field, f'Enter text between 1 and {limit} characters.')
    return value.strip()


def _unit(value, field):
    if not isinstance(value, str) or value not in {'m2', 'ha', 'acre'}:
        _error(field, 'Choose m2, ha or acre.')
    return value


def _geometry(value, field):
    try:
        return validate_polygon(value).as_geojson()
    except BoundaryValidationError as exc:
        raise CreationRequestError(str(exc), status=422, code=exc.code,
                                   fields={field: str(exc)}) from exc


def _revision(value):
    if 'registry_revision' not in value:
        raise CreationRequestError('Reload farm setup and supply registry_revision.',
                                   status=428, code='revision_required')
    revision = value['registry_revision']
    if type(revision) is not int or revision < 0:
        _error('registry_revision', 'Use the nonnegative integer revision from the registry read.')
    return revision


def _plot(value, prefix=''):
    if not isinstance(value, dict):
        _error(prefix.rstrip('.') or 'plot', 'Plot details and boundary are required.')
    _fields(value, {'name', 'crop_type', 'geometry', 'area_unit'}, prefix)
    crop = value.get('crop_type', '')
    if not isinstance(crop, str) or (crop and crop not in CROP_PARAMS):
        _error(prefix + 'crop_type', 'Choose a crop from the supported crop library.')
    return dict(name=_text(value.get('name'), prefix + 'name'), crop_type=crop,
                area_unit=_unit(value.get('area_unit', 'm2'), prefix + 'area_unit'),
                geometry=_geometry(value.get('geometry'), prefix + 'geometry'))


def parse_farm_creation(body):
    value = _parse(body)
    _fields(value, {'name', 'timezone', 'area_unit', 'owner', 'geometry',
                    'first_plot', 'registry_revision', 'latitude', 'longitude'})
    revision = _revision(value)
    timezone = _text(value.get('timezone'), 'timezone', 100)
    try:
        ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError):
        _error('timezone', 'Choose a valid IANA timezone.')
    owner = value.get('owner', '')
    if not isinstance(owner, str) or len(owner.strip()) > 200:
        _error('owner', 'Owner must be text up to 200 characters.')
    result = dict(name=_text(value.get('name'), 'name'), timezone_name=timezone,
        area_unit=_unit(value.get('area_unit', 'm2'), 'area_unit'), owner=owner.strip(),
        geometry=_geometry(value.get('geometry'), 'geometry'),
        first_plot=_plot(value.get('first_plot'), 'first_plot.'), registry_revision=revision)
    if ('latitude' in value) != ('longitude' in value):
        _error('latitude', 'Provide both reference latitude and longitude, or omit both.')
    for key, limit in (('latitude', 90), ('longitude', 180)):
        if key in value:
            number = value[key]
            try:
                valid = type(number) in (int, float) and math.isfinite(number) and abs(number) <= limit
            except OverflowError:
                valid = False
            if not valid:
                _error(key, 'Use a finite numeric reference coordinate within its valid range.')
            result[key] = number
    return result


def parse_plot_creation(body):
    value = _parse(body)
    _fields(value, {'name', 'crop_type', 'geometry', 'area_unit', 'registry_revision'})
    revision = _revision(value)
    plot = _plot({key: item for key, item in value.items() if key != 'registry_revision'})
    return dict(**plot, registry_revision=revision)


def validate_creation_boundary(body, registry):
    """Read-only preflight using the same geometry rules as the final save."""
    from geometry_validation import validate_farm_layout
    value = _parse(body)
    creating_farm = 'first_plot' in value
    request = parse_farm_creation(body) if creating_farm else parse_plot_creation(body)
    if request['registry_revision'] != registry['registry_revision']:
        raise CreationRequestError('The farm layout changed. Reload farm setup before continuing.',
                                   status=409, code='registry_revision_conflict')
    if creating_farm:
        if registry['farm'] is not None:
            raise CreationRequestError('A farm already exists. Reload farm setup.', status=409)
        farm = request['geometry']
        plots = {'draft': request['first_plot']['geometry']}
    else:
        if registry['farm'] is None:
            raise CreationRequestError('Create the farm first.', status=409)
        farm = registry['farm']['geometry']
        plots = {plot['plot_id']: plot['geometry'] for plot in registry['plots']}
        plots['draft'] = request['geometry']
    try:
        validate_farm_layout(farm, plots)
    except BoundaryValidationError as exc:
        messages = {
            'containment': 'Part of this plot is outside the farm boundary. Move it inside; shared edges are allowed.',
            'overlap': 'This plot overlaps an existing plot. Shared edges are allowed.',
        }
        message = messages.get(exc.code, str(exc))
        raise CreationRequestError(message, status=422, code=exc.code,
                                   fields={'geometry': message}) from exc
    return {'valid': True}
