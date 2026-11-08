"""Farmer-entered planting date invariant; satellite green-up is not a date source."""
from datetime import date
import re


def validate_planting_date(crop_type, value):
    if not crop_type:
        if value:
            raise ValueError('A planting date cannot be set for an unplanted plot.')
        return ''
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Enter the planting date for the selected crop.')
    value = value.strip()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Enter a valid planting date in YYYY-MM-DD format.')
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ValueError('Enter a valid planting date in YYYY-MM-DD format.') from None
    return value
