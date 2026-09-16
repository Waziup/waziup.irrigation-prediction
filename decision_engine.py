
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from crop_model import compute_gdd_from_weather, get_stress_threshold_series
from farm_config import FarmConfig
from phenology_engine import compute_kc_gdd

log = logging.getLogger(__name__)


# HORIZON LABEL PARSING

# Accepts: '24h', '48h', '6steps', 'horizon_48h'
_HORIZON_RE = re.compile(r"(\d+(?:\.\d+)?)(?:h|steps|)$")


def _parse_horizon_hours(label: str) -> float:
    """
    Extract the hour count from a horizon label string.

    Raises ValueError on unrecognised formats rather than silently
    returning 0 — a silent 0 would corrupt chronological sort and
    make all unknown horizons appear as time=0 breaches.
    """
    m = _HORIZON_RE.search(label)
    if not m:
        raise ValueError(
            f"Cannot parse horizon label {label!r}. "
            f"Expected a number at the end, e.g. '24h', '48h', '6steps'."
        )
    return float(m.group(1))


# DECISION OUTPUT STRUCTURES

@dataclass
class IrrigationRecommendation:
    """
    Output of the decision engine for a single evaluation point.

    urgency levels:
        'critical'  current tension reaches or exceeds threshold
        'advise'    breach forecast within advise_horizon_hours → act now
        'watch'     breach forecast but not imminent → monitor
        'none'      no breach in forecast window
        'error'     invalid input/forecast; fail-safe no irrigation
    """
    should_irrigate: bool
    urgency: str
    first_breach_horizon: Optional[str]           # 'now' when already breached
    first_breach_timestamp: Optional[datetime]
    current_tension: float                         # cbar
    stress_threshold: float                        # cbar
    forecast_summary: Dict[str, float]
    forecast_thresholds: Dict[str, float] = field(default_factory=dict)
    breach_horizons: List[str] = field(default_factory=list)  # chronological
    error_message: Optional[str] = None
    status_message: Optional[str] = None


def build_timestamp_forecast(
    predictions: pd.DataFrame,
    current_timestamp: Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    """Convert timestamped tension output into decision-engine horizons."""
    if (
        not isinstance(predictions, pd.DataFrame)
        or predictions.empty
        or "smoothed_values" not in predictions.columns
    ):
        return {}

    # Preserve every model-produced timestamp so changing model cadence also
    # changes decision horizons automatically.
    index = pd.DatetimeIndex(predictions.index).sort_values().unique()
    now = current_timestamp or pd.Timestamp.now(tz=index.tz)
    if index.tz is None and now.tzinfo is not None:
        now = now.tz_localize(None)
    elif index.tz is not None and now.tzinfo is None:
        now = now.tz_localize(index.tz)
    elif index.tz is not None and now.tzinfo is not None:
        now = now.tz_convert(index.tz)

    result = {}
    for timestamp in index[index > now]:
        value = predictions.loc[timestamp, "smoothed_values"]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        hours = (timestamp - now).total_seconds() / 3600.0
        label_hours = round(hours, 3)
        label = f"{int(label_hours)}h" if label_hours.is_integer(
        ) else f"{label_hours}h"
        result[label] = value
    return result


def error_recommendation(
    current_tension: float,
    stress_threshold: float,
    tension_forecast: Dict[str, float],
    message: str,
) -> IrrigationRecommendation:
    """Create a public fail-safe recommendation for unavailable evidence."""
    try:
        current = float(current_tension)
    except (TypeError, ValueError):
        current = float("nan")
    try:
        threshold = float(stress_threshold)
    except (TypeError, ValueError):
        threshold = float("nan")
    return IrrigationRecommendation(
        should_irrigate=False,
        urgency="error",
        first_breach_horizon=None,
        first_breach_timestamp=None,
        current_tension=current,
        stress_threshold=threshold,
        forecast_summary=dict(tension_forecast),
        breach_horizons=[],
        error_message=message,
    )


# Retain the private name for callers outside this repository that imported it.
_error_recommendation = error_recommendation


def inactive_season_recommendation(
    current_tension: float,
    stress_threshold: float,
    tension_forecast: Dict[str, float],
    message: str,
) -> IrrigationRecommendation:
    """Return an explicit no-irrigation decision outside the crop season."""
    return IrrigationRecommendation(
        should_irrigate=False,
        urgency="none",
        first_breach_horizon=None,
        first_breach_timestamp=None,
        current_tension=float(current_tension),
        stress_threshold=float(stress_threshold),
        forecast_summary=dict(tension_forecast),
        forecast_thresholds={},
        breach_horizons=[],
        status_message=message,
    )


def evaluate_forecast(
    current_tension: float,
    tension_forecast: Dict[str, float],
    stress_threshold: float,
    current_timestamp: Optional[datetime] = None,
    advise_horizon_hours: float = 24.0,
    watch_horizon_hours: float = 72.0,
    stress_thresholds: Optional[Dict[str, float]] = None,
) -> IrrigationRecommendation:
    """
    Decide whether to irrigate based on the current tension and a forecast.

    advise_horizon_hours and watch_horizon_hours should come from
    FarmConfig (which reads them from farms.yaml) so sensitive crops at
    critical stages get tighter horizons without hardcoding.

    Args:
        current_tension:       Current observed soil tension (cbar).
        tension_forecast:      Dict of {horizon_label: predicted_tension_cbar}.
                               Keys must be parseable, e.g. '24h', '48h'.
        stress_threshold:      Threshold applying to the current observation.
        stress_thresholds:     Optional threshold for every forecast horizon.
                               Required for a dynamic forecast decision.
        current_timestamp:     Used to compute first_breach_timestamp.
        advise_horizon_hours:  Breach within this window → 'advise' + irrigate.
        watch_horizon_hours:   Breach within this window → 'watch'.

    Returns:
        IrrigationRecommendation.
    """
    # Validate scalar inputs first so caller gets deterministic fail-safe output.
    try:
        current_tension = float(current_tension)
        stress_threshold = float(stress_threshold)
    except (TypeError, ValueError):
        return _error_recommendation(
            current_tension=np.nan,
            stress_threshold=np.nan,
            tension_forecast=tension_forecast,
            message="current_tension and stress_threshold must be numeric",
        )

    if not np.isfinite(current_tension) or not np.isfinite(stress_threshold):
        return _error_recommendation(
            current_tension=current_tension,
            stress_threshold=stress_threshold,
            tension_forecast=tension_forecast,
            message="current_tension and stress_threshold must be finite numbers",
        )

    if not isinstance(tension_forecast, dict):
        return _error_recommendation(
            current_tension=current_tension,
            stress_threshold=stress_threshold,
            tension_forecast={},
            message="tension_forecast must be a dict of {horizon_label: value}",
        )

    breach_horizons: List[str] = []
    first_breach_horizon: Optional[str] = None
    first_breach_timestamp: Optional[datetime] = None

    horizon_hours: Dict[str, float] = {}
    invalid_labels: List[str] = []
    invalid_values: List[str] = []
    valid_forecast: Dict[str, float] = {}
    valid_thresholds: Dict[str, float] = {}

    for label, value in tension_forecast.items():
        try:
            parsed_hours = _parse_horizon_hours(label)
        except ValueError:
            invalid_labels.append(str(label))
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            invalid_values.append(str(label))
            continue

        if not np.isfinite(numeric_value):
            invalid_values.append(str(label))
            continue

        horizon_hours[label] = parsed_hours
        valid_forecast[label] = numeric_value

    if stress_thresholds is not None:
        if not isinstance(stress_thresholds, dict):
            return _error_recommendation(
                current_tension, stress_threshold, tension_forecast,
                "stress_thresholds must be a dict keyed by forecast horizon")
        missing_thresholds = sorted(
            str(label) for label in valid_forecast if label not in stress_thresholds)
        invalid_thresholds = []
        for label in valid_forecast:
            if label not in stress_thresholds:
                continue
            if stress_thresholds[label] is None:
                continue
            try:
                value = float(stress_thresholds[label])
            except (TypeError, ValueError):
                invalid_thresholds.append(str(label))
                continue
            if not np.isfinite(value) or value <= 0:
                invalid_thresholds.append(str(label))
            else:
                valid_thresholds[label] = value
        if missing_thresholds or invalid_thresholds:
            return _error_recommendation(
                current_tension, stress_threshold, tension_forecast,
                "forecast thresholds are incomplete or invalid: "
                f"missing={missing_thresholds}, invalid={sorted(invalid_thresholds)}")
    else:
        valid_thresholds = {
            label: stress_threshold for label in valid_forecast
        }

    if invalid_labels or invalid_values:
        problems = []
        if invalid_labels:
            problems.append(f"invalid horizon labels={sorted(invalid_labels)}")
        if invalid_values:
            problems.append(
                f"non-numeric/non-finite values for={sorted(invalid_values)}")
        return _error_recommendation(
            current_tension=current_tension,
            stress_threshold=stress_threshold,
            tension_forecast=tension_forecast,
            message="; ".join(problems),
        )

    for label in sorted(valid_forecast.keys(), key=lambda k: horizon_hours[k]):
        if label not in valid_thresholds:
            # A null threshold explicitly marks an inactive crop-season
            # horizon (harvest or physiological maturity).
            continue
        if valid_forecast[label] >= valid_thresholds[label]:
            breach_horizons.append(label)
            if first_breach_horizon is None:
                first_breach_horizon = label
                if current_timestamp is not None:
                    first_breach_timestamp = current_timestamp + timedelta(
                        hours=horizon_hours[label]
                    )

    currently_breached = current_tension >= stress_threshold
    if currently_breached:
        # Prepend 'now' so breach_horizons is chronologically complete.
        breach_horizons.insert(0, "now")
        first_breach_horizon = "now"
        first_breach_timestamp = current_timestamp
        urgency = "critical"
        should_irrigate = True
    elif first_breach_horizon is not None:
        hours_to_breach = horizon_hours[first_breach_horizon]
        if hours_to_breach <= advise_horizon_hours:
            urgency = "advise"
            should_irrigate = True
        elif hours_to_breach <= watch_horizon_hours:
            urgency = "watch"
            should_irrigate = False
        else:
            urgency = "none"
            should_irrigate = False
    else:
        urgency = "none"
        should_irrigate = False

    return IrrigationRecommendation(
        should_irrigate=should_irrigate,
        urgency=urgency,
        first_breach_horizon=first_breach_horizon,
        first_breach_timestamp=first_breach_timestamp,
        current_tension=current_tension,
        stress_threshold=stress_threshold,
        forecast_summary=dict(valid_forecast),
        forecast_thresholds={
            label: valid_thresholds.get(label)
            for label in valid_forecast
        },
        breach_horizons=breach_horizons,
    )


# BATCH DECISION — vectorised over a DataFrame

def evaluate_forecast_series(
    predicted_tensions: pd.DataFrame,
    stress_thresholds: pd.Series,
) -> pd.DataFrame:
    """
    Vectorised breach detection across a DataFrame of tension forecasts.

    For each horizon column in predicted_tensions, adds:
        <horizon>_breached  bool   — predicted tension >= threshold
        <horizon>_margin    float  — predicted - threshold (negative = safe)

    stress_thresholds is forward-filled onto predicted_tensions.index so
    daily-resolution thresholds work correctly against sub-daily predictions.
    """
    thresholds = stress_thresholds.reindex(
        predicted_tensions.index, method="ffill")
    results = {}
    for col in predicted_tensions.columns:
        results[f"{col}_breached"] = predicted_tensions[col] >= thresholds
        results[f"{col}_margin"] = predicted_tensions[col] - thresholds
    return pd.DataFrame(results, index=predicted_tensions.index)


# OFFLINE EVALUATION HELPERS
def build_threshold_series_for_eval(
    df_test: pd.DataFrame,
    farm: FarmConfig,
) -> Optional[pd.Series]:
    """
    Compute the crop stress threshold series for an offline evaluation run.

    Reads all crop and soil parameters from FarmConfig.
    Returns None if Temperature is absent or planting_date is not set.

    Args:
        df_test:  Test DataFrame with DatetimeIndex and 'Temperature' column.
        farm:     FarmConfig for this plot.

    Returns:
        pd.Series of thresholds at df_test's frequency, or None on failure.
    """
    if "Temperature" not in df_test.columns:
        log.warning(
            "  [%s] build_threshold_series_for_eval: no Temperature column — cannot compute GDD-based thresholds",
            farm.farm_id,
        )
        return None

    daily = df_test["Temperature"].resample("1D")
    gdd_series = compute_gdd_from_weather(daily.max(), daily.min(), farm)
    etc_daily = None
    if str(getattr(farm, "threshold_mode", "static")) == "dynamic":
        if "Et0_evapotranspiration" not in df_test:
            log.warning(
                "  [%s] dynamic evaluation needs Et0_evapotranspiration",
                farm.farm_id)
            return None
        daily_et0 = pd.to_numeric(
            df_test["Et0_evapotranspiration"], errors="coerce").resample(
                "1D").sum(min_count=1)
        etc_daily = daily_et0.reindex(gdd_series.index) * gdd_series.apply(
            lambda value: compute_kc_gdd(value, farm.crop_type))
    daily_thresholds = get_stress_threshold_series(
        gdd_series, farm, etc_daily)
    return daily_thresholds.reindex(df_test.index, method="ffill")


def _fbeta(prec: float, rec: float, beta: float) -> float:
    """F-beta score. beta=2 weights recall 4× more than precision."""
    denom = beta**2 * prec + rec
    return (1 + beta**2) * prec * rec / denom if denom > 0 else 0.0


def _result_mae(item) -> float:
    """Helper to extract MAE from results_map items without loop-lambda capture."""
    return item[1]["mae"]


def evaluate_training_results(
    df_test: pd.DataFrame,
    forward_results: Dict,
    horizon_labels: Dict[int, str],
    target_col: str,
    farm: FarmConfig,
    f_beta: float = 2.0,
) -> Dict:
    """
    Evaluate tension forecast quality as an irrigation decision problem.

    Compares each horizon's predicted tension against the crop stress
    threshold to produce precision, recall, F-beta, and missed-irrigation
    metrics. Uses the current reading as a persistence baseline.

    Expected keys in each forward_results[h]['results'][model_name] dict:
        pred_abs    np.ndarray — absolute predicted tension values (cbar)
        test_index  pd.Index   — index of the prediction rows in df_test

    Metric choice — F-beta (default beta=2.0):
        Missed irrigation (false negative) costs more than false alarm
        (false positive) in crop production. beta=2 weights recall four
        times more than precision. F1 is also reported for benchmarking.

    Args:
        df_test:         Held-out test DataFrame.
        forward_results: Output of the training pipeline's horizon loop.
        horizon_labels:  {step_count: label_string}, e.g. {6: '6h'}.
        target_col:      Soil tension target column name — no default to
                         avoid silent mismatches ('grouped_soil' vs others).
        farm:            FarmConfig providing crop_type, planting_date,
                         soil_texture_class, advise/watch horizons.
        f_beta:          Beta for F-beta score. 2.0 = recall-weighted.

    Returns:
        Dict keyed by horizon step with decision quality metrics.
        Empty dict when prerequisites are missing.
    """
    thresholds = build_threshold_series_for_eval(df_test, farm)
    if thresholds is None:
        return {}

    print(f"\n{'─'*60}")
    print(f"  DECISION ENGINE EVALUATION  [{farm.farm_id}]")
    print(f"  crop={farm.crop_type}  planted={farm.planting_date}")
    print(f"  F-beta: beta={f_beta} (recall-weighted)")
    print(f"{'─'*60}")
    print(
        f"  Threshold: mean={thresholds.mean():.1f} std={thresholds.std():.1f} cbar")

    decision_results = {}

    for h, h_data in forward_results.items():
        if "results" not in h_data or not h_data["results"]:
            continue
        hl = horizon_labels.get(h, f"{h}steps")

        results_map = h_data["results"]
        best_name, best = min(results_map.items(), key=_result_mae)

        pred_abs = best.get("pred_abs")    # absolute predictions (cbar)
        pred_index = best.get("test_index")  # index into df_test

        if pred_abs is None or pred_index is None:
            log.warning(
                "  %s: missing 'pred_abs' or 'test_index' — check that train_farm_models.py stores these keys",
                hl,
            )
            continue

        # Coerce to ndarray — train_farm_models.py may store either a list
        # or an array; boolean masking and np.isnan require ndarray.
        pred_abs = np.asarray(pred_abs, dtype=float)

        y_current = df_test.loc[pred_index, target_col].values
        y_future = df_test[target_col].shift(-h).reindex(pred_index).values
        thresh_v = thresholds.reindex(pred_index).values

        valid = (
            ~np.isnan(y_future)
            & ~np.isnan(thresh_v)
            & ~np.isnan(pred_abs)
            # NaN current reading → unknown persist baseline
            & ~np.isnan(y_current)
        )
        if valid.sum() < 20:
            continue

        pred_v = pred_abs[valid]
        future_v = y_future[valid]
        thresh_v = thresh_v[valid]
        current_v = y_current[valid]

        actual_breach = future_v > thresh_v
        predicted_breach = pred_v > thresh_v
        persist_breach = current_v > thresh_v

        n_total = len(actual_breach)
        n_actual_breach = int(actual_breach.sum())

        if n_actual_breach == 0 or n_actual_breach == n_total:
            label = "all breached" if n_actual_breach == n_total else "all safe"
            print(f"  {hl}: Skipping — {label} ({n_actual_breach}/{n_total})")
            continue

        tp = int((predicted_breach & actual_breach).sum())
        fp = int((predicted_breach & ~actual_breach).sum())
        fn = int((~predicted_breach & actual_breach).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = _fbeta(precision, recall, 1.0)
        fb = _fbeta(precision, recall, f_beta)

        tp_p = int((persist_breach & actual_breach).sum())
        fp_p = int((persist_breach & ~actual_breach).sum())
        fn_p = int((~persist_breach & actual_breach).sum())
        prec_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0.0
        rec_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0.0
        fb_p = _fbeta(prec_p, rec_p, f_beta)
        f1_p = _fbeta(prec_p, rec_p, 1.0)

        missed_rate = fn / n_total
        false_alarm_rate = fp / n_total
        beat = " BEATS PERSISTENCE" if fb > fb_p else ""

        print(
            f"  {hl} ({best_name}):  F{f_beta:.0f}={fb:.3f} vs persist={fb_p:.3f}{beat}")
        print(f"    F1={f1:.3f}  P={precision:.3f}  R={recall:.3f}  "
              f"Missed={missed_rate:.1%}  FalseAlarm={false_alarm_rate:.1%}  "
              f"BreachRate={n_actual_breach/n_total:.1%}")

        decision_results[h] = {
            "horizon_label":          hl,
            "model":                  best_name,
            f"f{f_beta:.0f}":         fb,
            "f1":                     f1,
            "precision":              precision,
            "recall":                 recall,
            f"f{f_beta:.0f}_persist": fb_p,
            "f1_persistence":         f1_p,
            "missed_irrigation_rate": missed_rate,
            "false_alarm_rate":       false_alarm_rate,
            "breach_rate":            n_actual_breach / n_total,
            "n_test":                 n_total,
        }

    return decision_results
