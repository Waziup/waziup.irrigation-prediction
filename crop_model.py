import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from crops import (
    STAGE_NAMES,
    IRRIGATION_TRIGGER_BASE_BY_TEXTURE,
    IRRIGATION_TRIGGER_BASE_DEFAULT,
    STAGE_PRE_EMERGENCE,
    get_crop_params,
)
from farm_config import FarmConfig
from phenology_engine import (
    compute_gdd_series,
    compute_kc_dynamic,
    compute_kc_gdd_series,
    compute_kc_ndvi,
    compute_kc_ndre,
    compute_kc_dynamic_series,
    get_dynamic_threshold,
    get_dynamic_threshold_series,
    get_growth_stage,
    get_growth_stage_series,
    sar_adjusted_ndvi_age,
)

log = logging.getLogger(__name__)


@dataclass
class CropState:

    growth_stage: int
    growth_stage_name: str
    stress_threshold_cbar: float    # cbar — irrigate when predicted tension exceeds this
    kc: float                       # blended Kc_dynamic
    gdd_cumulative: float
    crop_type: str
    # crop water demand (mm/day); 0.0 when ET0 unavailable
    etc_daily_mm: float = 0.0
    recommended_volume_mm: Optional[float] = None
    recommended_volume_m3: Optional[float] = None
    satellite_validation: Optional[dict] = None
    et0_today_mm: Optional[float] = None
    et0_baseline_mm: Optional[float] = None
    et0_std_mm: Optional[float] = None
    # Per-index source age/quality are kept separate because NDRE cadence can
    # differ materially from NDVI cadence in the EO catalog.
    sat_ndvi_quality: float = 0.0
    sat_ndre_quality: float = 0.0
    sat_ndre_age_hours: float = float("inf")


def get_crop_state(
    farm: FarmConfig,
    gdd_cumulative: float,
    ndvi: float = float("nan"),
    ndre: float = float("nan"),
    ndvi_age_hours: float = float("inf"),
    ndre_age_hours: float = float("inf"),
    ndvi_quality: Optional[float] = None,
    ndre_quality: Optional[float] = None,
    etc_daily_mm: float = 0.0,
) -> CropState:
    crop_type = farm.crop_type
    soil_texture_class = farm.soil_texture_class

    stage = get_growth_stage(gdd_cumulative, crop_type)
    trigger_base = IRRIGATION_TRIGGER_BASE_BY_TEXTURE.get(
        soil_texture_class, IRRIGATION_TRIGGER_BASE_DEFAULT)
    threshold = get_dynamic_threshold(gdd_cumulative, crop_type, trigger_base)
    kc = compute_kc_dynamic(
        cumulative_gdd=gdd_cumulative,
        ndvi=ndvi,
        ndvi_age_hours=ndvi_age_hours,
        crop_type=crop_type,
        ndre=ndre,
        ndre_age_hours=ndre_age_hours,
        ndvi_quality=ndvi_quality,
        ndre_quality=ndre_quality,
    )
    return CropState(
        growth_stage=stage,
        growth_stage_name=STAGE_NAMES[stage],
        stress_threshold_cbar=threshold,
        kc=kc,
        gdd_cumulative=gdd_cumulative,
        crop_type=crop_type,
        etc_daily_mm=etc_daily_mm,
        recommended_volume_mm=etc_daily_mm,
        recommended_volume_m3=None,
        sat_ndvi_quality=(float(ndvi_quality)
                          if ndvi_quality is not None and np.isfinite(ndvi_quality) else 0.0),
        sat_ndre_quality=(float(ndre_quality)
                          if ndre_quality is not None and np.isfinite(ndre_quality) else 0.0),
        sat_ndre_age_hours=float(ndre_age_hours),
    )


def get_stress_threshold(
    farm: FarmConfig,
    gdd_cumulative: float,
) -> float:
    trigger_base = IRRIGATION_TRIGGER_BASE_BY_TEXTURE.get(
        farm.soil_texture_class, IRRIGATION_TRIGGER_BASE_DEFAULT)
    return get_dynamic_threshold(gdd_cumulative, farm.crop_type, trigger_base)


def get_stress_threshold_series(
    gdd_series: pd.Series,
    farm: FarmConfig,
) -> pd.Series:
    trigger_base = IRRIGATION_TRIGGER_BASE_BY_TEXTURE.get(
        farm.soil_texture_class, IRRIGATION_TRIGGER_BASE_DEFAULT)
    return get_dynamic_threshold_series(gdd_series, farm.crop_type, trigger_base)


def compute_gdd_from_weather(
    tmax_series: pd.Series,
    tmin_series: pd.Series,
    farm: FarmConfig,
) -> pd.Series:
    return compute_gdd_series(
        tmax_series,
        tmin_series,
        farm.planting_date,
        farm.crop_type,
        farm.initial_gdd,
    )


def compute_phenology_features(
    df: pd.DataFrame,
    farm: FarmConfig,
) -> pd.DataFrame:
    df = df.copy()
    crop_type = farm.crop_type
    params = get_crop_params(crop_type)

    planting_ts = pd.Timestamp(farm.planting_date)
    if df.index.tz is not None and planting_ts.tz is None:
        planting_ts = planting_ts.tz_localize(df.index.tz)
    elif df.index.tz is None and planting_ts.tz is not None:
        planting_ts = planting_ts.tz_localize(None)

    if "Temperature" in df.columns:
        daily_tmax = df["Temperature"].resample("D").max()
        daily_tmin = df["Temperature"].resample("D").min()
    elif "Tmax_daily" in df.columns and "Tmin_daily" in df.columns:
        daily_tmax = df["Tmax_daily"].resample("D").first()
        daily_tmin = df["Tmin_daily"].resample("D").first()
        log.info("  Using Tmax_daily/Tmin_daily for GDD")
    else:
        log.error(
            "  Farm '%s': no temperature data found — need 'Temperature' or 'Tmax_daily'/'Tmin_daily'. "
            "Phenology columns set to zero/NaN defaults.",
            farm.farm_id,
        )
        df["gdd_cumulative"] = 0.0
        df["growth_stage"] = STAGE_PRE_EMERGENCE
        df["kc_gdd"] = params.kc_ini
        df["kc_ndvi"] = np.nan
        df["kc_ndre"] = np.nan
        df["kc_dynamic"] = params.kc_ini
        df["etc_daily"] = 0.0
        df["dynamic_threshold"] = IRRIGATION_TRIGGER_BASE_DEFAULT
        return df

    log.info(
        "  [%s] GDD: crop=%s, T_base=%.1f°C, T_ceil=%.1f°C, planting=%s",
        farm.farm_id,
        crop_type,
        params.t_base,
        params.t_ceiling,
        planting_ts.date(),
    )
    daily_gdd = compute_gdd_series(
        daily_tmax, daily_tmin, planting_ts, crop_type,
        initial_gdd=farm.initial_gdd,
    )
    gdd_hourly = daily_gdd.reindex(df.index, method="ffill")
    n_unfilled = int(gdd_hourly.isna().sum())
    if n_unfilled > 0:
        log.warning(
            "  %s hourly rows before first daily GDD — filling with 0",
            n_unfilled,
        )
    df["gdd_cumulative"] = gdd_hourly.fillna(0.0)

    df["growth_stage"] = get_growth_stage_series(
        df["gdd_cumulative"], crop_type)
    gdd_final = df["gdd_cumulative"].iloc[-1]
    log.info("  GDD range: 0 → %.0f", gdd_final)
    for s in sorted(df["growth_stage"].unique()):
        first_idx = df[df["growth_stage"] == s].index[0]
        log.info(
            "    %s entered %s (GDD=%.0f)",
            f"{STAGE_NAMES[s]:20s}",
            first_idx.date(),
            df.loc[first_idx, 'gdd_cumulative'],
        )

    df["kc_gdd"] = compute_kc_gdd_series(df["gdd_cumulative"], crop_type)

    df["kc_ndvi"] = (
        compute_kc_ndvi(df["sat_ndvi"].values.astype(float), crop_type)
        if "sat_ndvi" in df.columns else np.nan
    )
    df["kc_ndre"] = (
        compute_kc_ndre(df["sat_ndre"].values.astype(float), crop_type)
        if "sat_ndre" in df.columns else np.nan
    )

    if "sat_ndvi_age" in df.columns:
        ndvi_age = df["sat_ndvi_age"].fillna(np.inf)
        log.info(
            "  sat_ndvi_age: mean=%.1fh",
            ndvi_age[ndvi_age < np.inf].mean(),
        )
    elif "satellite_data_age" in df.columns:
        ndvi_age = df["satellite_data_age"].fillna(np.inf)
        log.info("  Using satellite_data_age as NDVI age proxy")
    else:
        ndvi_age = pd.Series(np.inf, index=df.index)
        log.warning(
            "  No satellite age — Kc_dynamic will use pure GDD-based Kc")

    # SAR correction: when VV backscatter confirms canopy, cap NDVI staleness
    # so cloud-cover gaps don't decay the satellite quality weight to zero.
    if "sat_vv_db" in df.columns:
        ndvi_age = sar_adjusted_ndvi_age(ndvi_age, df["sat_vv_db"])

    ndvi_col = (df["sat_ndvi"] if "sat_ndvi" in df.columns
                else pd.Series(np.nan, index=df.index))
    ndre_col = df["sat_ndre"] if "sat_ndre" in df.columns else None
    df["kc_dynamic"] = compute_kc_dynamic_series(
        df["gdd_cumulative"], ndvi_col, ndvi_age, crop_type, ndre_col,
        df.get("sat_ndvi_quality"), df.get("sat_ndre_age"),
        df.get("sat_ndre_quality"),
    )

    if "Et0_evapotranspiration" in df.columns:
        df["etc_daily"] = (
            (df["Et0_evapotranspiration"] * df["kc_dynamic"])
            .rolling(24, min_periods=1).sum()
        )
    else:
        df["etc_daily"] = 0.0

    trigger_base = IRRIGATION_TRIGGER_BASE_BY_TEXTURE.get(
        farm.soil_texture_class, IRRIGATION_TRIGGER_BASE_DEFAULT)
    df["dynamic_threshold"] = get_dynamic_threshold_series(
        df["gdd_cumulative"], crop_type, trigger_base
    )

    log.info(
        "  Kc_dynamic [%.3f, %.3f]  ETc [%.2f, %.2f] mm/day  threshold [%.1f, %.1f] cbar",
        df["kc_dynamic"].min(),
        df["kc_dynamic"].max(),
        df["etc_daily"].min(),
        df["etc_daily"].max(),
        df["dynamic_threshold"].min(),
        df["dynamic_threshold"].max(),
    )
    return df
