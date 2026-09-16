"""Crop water requirement calculations with explicit units.

All depths are millimetres and all plot-level volumes are cubic metres.  Weather
conditions such as temperature, humidity, radiation and wind are represented by
reference evapotranspiration (ET0); crop type and stage are represented by Kc.
Rainfall reduces the irrigation requirement only by its effective fraction.
"""

from __future__ import annotations

import math
from typing import Optional


DEFAULT_APPLICATION_EFFICIENCY = 0.85
DEFAULT_EFFECTIVE_RAINFALL_FRACTION = 0.80


def _finite_non_negative(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _fraction(value, name: str) -> float:
    number = _finite_non_negative(value, name)
    if number <= 0 or number > 1:
        raise ValueError(f"{name} must be greater than 0 and no greater than 1")
    return number


def depth_mm_to_volume_m3(depth_mm: float, area_m2: float) -> float:
    """Convert a water depth over a plot into volume (1 mm = 1 L/m2)."""
    depth = _finite_non_negative(depth_mm, "depth_mm")
    area = _finite_non_negative(area_m2, "area_m2")
    if area <= 0:
        raise ValueError("area_m2 must be greater than 0")
    return depth * area / 1000.0


def volume_m3_to_depth_mm(volume_m3: float, area_m2: float) -> float:
    """Convert an applied plot volume to its equivalent water depth."""
    volume = _finite_non_negative(volume_m3, "volume_m3")
    area = _finite_non_negative(area_m2, "area_m2")
    if area <= 0:
        raise ValueError("area_m2 must be greater than 0")
    return volume * 1000.0 / area


def calculate_crop_water_requirement(
    *,
    et0_mm: float,
    kc: float,
    rainfall_mm: float,
    application_efficiency: float = DEFAULT_APPLICATION_EFFICIENCY,
    effective_rainfall_fraction: float = DEFAULT_EFFECTIVE_RAINFALL_FRACTION,
    area_m2: Optional[float] = None,
) -> dict:
    """Calculate ETc, effective rain, net/gross demand and optional volume.

    This calculates the irrigation needed to meet crop evapotranspiration over
    the supplied period. Existing soil water influences *when* irrigation is
    required through the soil-moisture/tension model. Recorded irrigation since
    the preceding model calculation is credited later, when the recommendation
    contract converts this gross requirement into a remaining amount.
    """
    et0 = _finite_non_negative(et0_mm, "et0_mm")
    coefficient = _finite_non_negative(kc, "kc")
    rainfall = _finite_non_negative(rainfall_mm, "rainfall_mm")
    efficiency = _fraction(application_efficiency, "application_efficiency")
    rain_fraction = _fraction(
        effective_rainfall_fraction, "effective_rainfall_fraction")

    etc = et0 * coefficient
    result = calculate_irrigation_requirement_from_etc(
        etc_mm=etc,
        rainfall_mm=rainfall,
        application_efficiency=efficiency,
        effective_rainfall_fraction=rain_fraction,
        area_m2=area_m2,
    )

    return {
        "et0_mm": et0,
        "kc": coefficient,
        **result,
    }


def calculate_irrigation_requirement_from_etc(
    *,
    etc_mm: float,
    rainfall_mm: float,
    application_efficiency: float = DEFAULT_APPLICATION_EFFICIENCY,
    effective_rainfall_fraction: float = DEFAULT_EFFECTIVE_RAINFALL_FRACTION,
    area_m2: Optional[float] = None,
) -> dict:
    """Calculate net and gross irrigation after ETc has been accumulated."""
    etc = _finite_non_negative(etc_mm, "etc_mm")
    rainfall = _finite_non_negative(rainfall_mm, "rainfall_mm")
    efficiency = _fraction(application_efficiency, "application_efficiency")
    rain_fraction = _fraction(
        effective_rainfall_fraction, "effective_rainfall_fraction")
    effective_rain = rainfall * rain_fraction
    net = max(0.0, etc - effective_rain)
    gross = net / efficiency

    volume = None
    if area_m2 is not None:
        volume = depth_mm_to_volume_m3(gross, area_m2)

    return {
        "etc_mm": etc,
        "rainfall_mm": rainfall,
        "effective_rainfall_mm": effective_rain,
        "effective_rainfall_fraction": rain_fraction,
        "net_irrigation_mm": net,
        "application_efficiency": efficiency,
        "gross_irrigation_mm": gross,
        "volume_m3": volume,
    }
