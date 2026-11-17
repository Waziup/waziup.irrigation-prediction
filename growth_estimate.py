"""Read-only crop growth and forecast-period demand, independent of tension model."""
from types import SimpleNamespace
import math
import pandas as pd
from phenology_engine import (daily_temperature_extrema, compute_gdd_series,
                              compute_kc_gdd, get_growth_stage, phenology_season_start)
from crops import STAGE_NAMES
from crop_model import get_crop_season_status
from water_demand import calculate_crop_water_requirement
from runtime_config import DEFAULT_FORECAST_HORIZON_DAYS
from weather_quality import deduplicate_weather_frame, complete_daily_et0


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _frame(frame):
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors='coerce')
    return deduplicate_weather_frame(frame[~frame.index.isna()].sort_index())


def build_growth_estimate(plot, *, fetch_weather, latitude, longitude,
                          timezone_name='UTC', store=None, crop_state=None, now=None):
    now = pd.Timestamp(now or pd.Timestamp.now(tz='UTC'))
    now = now.tz_localize(
        'UTC') if now.tzinfo is None else now.tz_convert('UTC')
    local_now = now.tz_convert(timezone_name)
    days = _finite(getattr(plot, 'forecast_horizon_days',
                   DEFAULT_FORECAST_HORIZON_DAYS))
    period_hours = days * \
        24 if days is not None and days > 0 else DEFAULT_FORECAST_HORIZON_DAYS * 24
    end = now + pd.Timedelta(hours=period_hours)
    result = dict(current_gdd=None, current_stage=None, kc_today=None,
                  etc_forecast_mm=None, etc_today_mm=None, growth_reason=None, evaluated_at=now.isoformat())
    demand = dict(available=False, reason=None, period_hours=period_hours,
                  period_start=now.isoformat(), period_end=end.isoformat(),
                  et0_mm=None, etc_mm=None, rainfall_mm=None, volume_m3=None,
                  credited_volume_m3=None, remaining_volume_m3=None,
                  credit_since=None, credit_until=now.isoformat(),
                  automatic_irrigation_allowed=False)
    result['water_demand'] = demand
    crop = getattr(plot, 'crop_type', '')
    planting = getattr(plot, 'planting_date', None)
    if not crop or not planting:
        result['growth_reason'] = demand['reason'] = 'Select a crop and planting date in Plot settings.'
        return result
    if _finite(latitude) is None or _finite(longitude) is None:
        result['growth_reason'] = demand['reason'] = 'Set the farm location to retrieve weather.'
        return result
    try:
        planting_at = pd.Timestamp(planting)
        planting_at = (planting_at.tz_localize(timezone_name) if planting_at.tzinfo is None
                       else planting_at.tz_convert(timezone_name))
    except (ValueError, TypeError):
        result['growth_reason'] = demand['reason'] = 'Set a valid planting date in Plot settings.'
        return result
    if planting_at > local_now:
        result['growth_reason'] = demand['reason'] = 'The crop has not been planted yet.'
        return result
    # Prioritise the short forecast window before opening a season of history.
    forecast = None
    forecast_error = None
    try:
        forecast = _frame(fetch_weather(latitude, longitude,
                                        start_date=local_now.normalize().tz_convert('UTC').strftime('%Y-%m-%d'),
                                        end_date=end.ceil('h').strftime('%Y-%m-%d')))
        daily_et0 = complete_daily_et0(forecast, timezone_name)
        day_total = _finite(daily_et0.get(local_now.normalize()))
        if day_total is not None:
            result['et0_today_mm'] = day_total
    except Exception as exc:
        forecast_error = exc
    try:
        season_start = phenology_season_start(local_now, planting_at, crop)
        # Read through the canonical weather cache even if a stopped/failed
        # tension pipeline still holds an older crop-state frame.
        history = fetch_weather(latitude, longitude,
                                start_date=season_start.tz_convert(
                                    'UTC').strftime('%Y-%m-%d'),
                                end_date=now.strftime('%Y-%m-%d'))
        history = _frame(history)
        temperatures = pd.to_numeric(history.get(
            'Temperature', pd.Series(dtype=float)), errors='coerce')
        high, low = daily_temperature_extrema(
            temperatures, timezone_name, through=local_now)
        expected = pd.date_range(
            season_start.normalize(), local_now.normalize(), freq='D')
        initial = float(getattr(plot, 'initial_gdd', 0) or 0)
        if initial > 0 and len(high):
            expected = expected[expected >= high.index.min()]
        high, low = high.reindex(expected), low.reindex(expected)
        if not len(high) or high.isna().any() or low.isna().any():
            raise ValueError(
                'Temperature history is incomplete for calculating GDD.')
        gdd = float(compute_gdd_series(high, low, planting,
                    crop, initial_gdd=initial).iloc[-1])
        if not math.isfinite(gdd) or gdd < 0:
            raise ValueError(
                'GDD could not be calculated from the available temperature history.')
        kc = float(compute_kc_gdd(gdd, crop))
        # Preserve optional EO refinement only when it belongs to this GDD state.
        if crop_state is not None and abs(float(getattr(crop_state, 'gdd_cumulative', -1)) - gdd) < .01:
            refined = _finite(getattr(crop_state, 'kc', None))
            if refined is not None and refined >= 0:
                kc = refined
        result.update(current_gdd=gdd, current_stage=STAGE_NAMES[get_growth_stage(
            gdd, crop)], kc_today=kc)
        if result.get('et0_today_mm') is not None:
            result['etc_today_mm'] = result['et0_today_mm'] * kc
    except Exception as exc:
        result['growth_reason'] = demand['reason'] = str(exc)
        return result
    try:
        if forecast_error is not None:
            raise forecast_error
        if forecast is None:
            raise ValueError('Forecast weather is unavailable.')
        # Provider rows are hourly accumulation endpoints. Prorate the first
        # and last intervals; gaps remain missing rather than implying zero.
        endpoints = pd.date_range(
            now.floor('h') + pd.Timedelta(hours=1), end.ceil('h'), freq='h')
        hourly = forecast.reindex(endpoints)
        fractions = pd.Series([
            (min(t, end) - max(t - pd.Timedelta(hours=1), now)).total_seconds()/3600
            for t in endpoints], index=endpoints)
        et0 = pd.to_numeric(hourly.get('Et0_evapotranspiration', pd.Series(
            index=endpoints, dtype=float)), errors='coerce')
        rain = pd.to_numeric(hourly.get('Rain', pd.Series(
            index=endpoints, dtype=float)), errors='coerce')
        # ETc is still useful if rain or plot area is missing.
        if et0.notna().all() and (et0 >= 0).all():
            demand['et0_mm'] = float((et0 * fractions).sum())
            result['etc_forecast_mm'] = demand['et0_mm'] * kc
            demand['etc_mm'] = result['etc_forecast_mm']
        if demand['et0_mm'] is None or rain.isna().any() or (rain < 0).any():
            raise ValueError(
                'Complete forecast ET0 and rainfall are required for the full forecast period.')
        season = get_crop_season_status(SimpleNamespace(crop_type=crop,
                                                        planting_date=planting, harvest_date=getattr(plot, 'harvest_date', None)), gdd, as_of=local_now)
        if not season['active']:
            raise ValueError(
                'Estimated irrigation requirement is unavailable outside the active crop season.')
        area = _finite(getattr(plot, 'plot_area_m2', None))
        requirement = calculate_crop_water_requirement(et0_mm=demand['et0_mm'], kc=kc,
                                                       rainfall_mm=float(
                                                           (rain * fractions).sum()),
                                                       application_efficiency=getattr(
            plot, 'application_efficiency', .85),
            effective_rainfall_fraction=getattr(
                plot, 'effective_rainfall_fraction', .8),
            area_m2=area if area is not None and area > 0 else None)
        demand.update(requirement)
        if demand['volume_m3'] is None:
            raise ValueError(
                'Set a positive plot area to estimate water volume.')
        pipeline = getattr(plot, 'pipeline_result', None)
        since = (getattr(pipeline, 'previous_calculated_at', None)
                 or getattr(pipeline, 'calculated_at', None) or local_now.normalize())
        if store is None:
            raise ValueError('Recorded irrigation could not be checked.')
        credit = store.applied_irrigation_since(plot.stable_id, since, now)
        credited = min(demand['volume_m3'], max(
            0., float(credit['volume_m3'])))
        demand.update(available=True, credited_volume_m3=credited,
                      remaining_volume_m3=max(
                          0., demand['volume_m3']-credited),
                      credit_since=credit['since'], credit_until=credit['until'])
    except Exception as exc:
        demand['reason'] = str(exc)
    return result
