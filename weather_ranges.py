"""Weather history is farm-wide and capped at 30 completed local days."""
from datetime import timedelta
import math


def resolve_weather_range(plot, view, selected, today):
    if view == 'forecast':
        if selected not in ('model', 'today'):
            raise ValueError('Choose a forecast time range.')
        horizon = float(getattr(plot, 'forecast_horizon_days', 5) or 5)
        if not math.isfinite(horizon) or horizon <= 0:
            raise ValueError('Check the prediction window in Plot settings.')
        end = today if selected == 'today' else today + timedelta(days=math.ceil(horizon))
        return today, end
    if view != 'historical' or selected != 'recent':
        raise ValueError('Choose Forecast or History.')
    return today - timedelta(days=30), today - timedelta(days=1)
