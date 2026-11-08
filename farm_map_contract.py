"""Read-only spatial projection of the existing farm dashboard contract."""
from copy import deepcopy
import json

from geometry_validation import validate_farm_layout


class FarmMapContractError(ValueError):
    """Stored spatial data or dashboard identity is inconsistent."""


def build_farm_map(*, registry, dashboard=None):
    """No status calculations, writes, tile dependencies or synthetic boundaries."""
    records = registry['plots']
    base = {'schema_version': '1.0',
            'registry_revision': registry['registry_revision']}
    if registry['farm'] is None:
        if records:
            raise FarmMapContractError('Plots cannot exist without a farm.')
        return dict(base, setup={'complete': False, 'reason': 'no_farm'},
                    farm=None, plots=[], summary=None, generated_at=None)
    if dashboard is None or not records:
        raise FarmMapContractError('A configured farm requires plots and dashboard data.')
    ids = [record['plot_id'] for record in records]
    cards = dashboard['plots']
    card_ids = [card['plot_id'] for card in cards]
    if len(set(ids)) != len(ids) or len(set(card_ids)) != len(card_ids) or set(ids) != set(card_ids):
        raise FarmMapContractError('Map and dashboard plot identities must match.')
    # Revalidate the whole layout: never hide corrupt or unmapped records.
    farm_boundary, boundaries = validate_farm_layout(
        registry['farm']['geometry'], {p['plot_id']: p['geometry'] for p in records})

    def spatial(record, boundary):
        version = record.get('geometry_version')
        updated = record.get('geometry_updated_at')
        if not isinstance(version, str) or not version or not isinstance(updated, str) or not updated:
            raise FarmMapContractError('Saved boundary metadata is required.')
        return {'geometry': boundary.as_geojson(), 'area_m2': boundary.area_m2,
                'geometry_version': version, 'geometry_updated_at': updated}

    by_id = {card['plot_id']: card for card in cards}
    plots = []
    for record in records:
        card = deepcopy(by_id[record['plot_id']])
        card.update(spatial(record, boundaries[record['plot_id']]))
        # Persisted crop remains available even before a prediction cycle.
        card['crop_type'] = card.get('crop_type') or record.get('crop_type')
        plots.append(card)
    farm = deepcopy(dashboard['farm'])
    farm.update(spatial(registry['farm'], farm_boundary))
    payload = dict(base, setup={'complete': True, 'reason': None}, farm=farm,
                   plots=plots, summary=deepcopy(dashboard['summary']),
                   generated_at=dashboard['generated_at'])
    # Reject malformed/non-finite response data before the HTTP JSON sanitizer
    # could disguise it as null. No raw runtime objects cross the boundary.
    json.dumps(payload, allow_nan=False)
    return payload
