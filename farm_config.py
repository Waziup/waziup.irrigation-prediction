import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

@dataclass
class FarmConfig:
    """
    Configuration for a single farm or plot.

    crop_type and planting_date are REQUIRED. Absence raises ValueError at
    construction time so failures are immediate and unambiguous.

    NOT frozen: sensor column categorization (tension_sensor_cols, etc.) is
    populated at runtime during CSV parsing and must be mutable. The core config
    (crop_type, planting_date, etc.) is still validated immutably in __post_init__.

    Runtime agronomic values are resolved from plot settings stored in SQLite.
    """
    name: str
    site_id: int

    csv_path: str

    latitude: float
    longitude: float
    timezone: str
    sensor_kind: str

    enable_experimental_ndre_kc: bool = False
    irrigation_type: str = ""
    irrigation_mode: str = ""
    application_efficiency: float = 0.85
    effective_rainfall_fraction: float = 0.80
    # Device identifiers may originate in the compatibility JSON layer, but
    # YAML values override them when explicitly configured.
    moisture_sensor_ids: List[str] = field(default_factory=list)
    temperature_sensor_ids: List[str] = field(default_factory=list)
    flow_sensor_ids: List[str] = field(default_factory=list)
    flow_confirmation_sensor_ids: List[str] = field(default_factory=list)
    flow_confirmation_mode: str = "event"

    crop_type: str = ""
    planting_date: str = ""
    harvest_date: Optional[str] = None

    soil_texture_class: Optional[int] = None   # USDA 0-11
    static_threshold_cbar: Optional[float] = None
    threshold_mode: str = "static"

    # Dynamic tension threshold inputs. VWC values are fractions (m3/m3).
    field_capacity_vwc: Optional[float] = None
    wilting_point_vwc: Optional[float] = None
    root_depth_m: Optional[float] = None
    sensor_depth_m: Optional[float] = None
    depletion_fraction: Optional[float] = None
    threshold_hysteresis_cbar: float = 0.0
    stage_depletion_fractions: Dict = field(default_factory=dict)
    stage_thresholds_cbar: Dict = field(default_factory=dict)

    field_capacity_lower: Optional[float] = None
    permanent_wilting_point: Optional[float] = None
    saturation: float = 0.0
    soil_calibration: Dict = field(default_factory=dict)
    soil_water_retention_curve: List = field(default_factory=list)
    plot_area_m2: Optional[float] = None

    initial_gdd: float = 0.0

    advise_horizon_hours: float = 24.0
    watch_horizon_hours:  float = 72.0
    # All scheduling and forecast controls are validated with the farm record.
    forecast_interval_minutes: int = 60
    forecast_horizon_days: float = 5.0
    predict_period_hours: float = 3.0
    retraining_interval_days: float = 1.0
    irrigation_confirmation_seconds: int = 10800
    error_retry_seconds: int = 1800

    # Runtime-populated sensor column categorization (tension/capacitive/converted only)
    tension_sensor_cols:  List[str] = field(default_factory=list)
    moisture_sensor_cols: List[str] = field(default_factory=list)
    temp_sensor_cols:     List[str] = field(default_factory=list)
    battery_cols:         List[str] = field(default_factory=list)
    resistance_cols:      List[str] = field(default_factory=list)
    other_cols:           List[str] = field(default_factory=list)

    def __post_init__(self):
        # Fail during startup rather than running with unsafe agronomic defaults.
        if not self.timezone:
            raise ValueError(f"Farm '{self.name}': timezone is required")
        if self.sensor_kind not in {"tension", "capacitive", "both"}:
            raise ValueError(
                f"Farm '{self.name}': unsupported sensor_kind '{self.sensor_kind}'"
            )
        if self.irrigation_mode and self.irrigation_mode not in {
                "automatic", "approval_required", "manual", "advisory_only"}:
            raise ValueError(
                f"Farm '{self.name}': unsupported irrigation_mode "
                f"'{self.irrigation_mode}'"
            )
        if self.flow_confirmation_mode not in {"event", "cumulative"}:
            raise ValueError(
                f"Farm '{self.name}': flow_confirmation_mode must be "
                "'event' or 'cumulative'")
        if not math.isfinite(self.latitude) or not -90 <= self.latitude <= 90:
            raise ValueError(
                f"Farm '{self.name}': latitude must be between -90 and 90")
        if not math.isfinite(self.longitude) or not -180 <= self.longitude <= 180:
            raise ValueError(
                f"Farm '{self.name}': longitude must be between -180 and 180")
        if self.advise_horizon_hours < 0 or self.watch_horizon_hours < 0:
            raise ValueError(
                f"Farm '{self.name}': decision horizons cannot be negative"
            )
        if self.forecast_interval_minutes <= 0:
            raise ValueError(
                f"Farm '{self.name}': forecast_interval_minutes must be positive"
            )
        if self.forecast_horizon_days <= 0:
            raise ValueError(
                f"Farm '{self.name}': forecast_horizon_days must be positive"
            )
        if self.predict_period_hours <= 0:
            raise ValueError(
                f"Farm '{self.name}': predict_period_hours must be positive"
            )
        if (self.retraining_interval_days <= 0
                or self.irrigation_confirmation_seconds <= 0
                or self.error_retry_seconds <= 0):
            raise ValueError(
                f"Farm '{self.name}': retraining and retry intervals must be positive"
            )
        for name, value in (
                ("application_efficiency", self.application_efficiency),
                ("effective_rainfall_fraction", self.effective_rainfall_fraction)):
            if not math.isfinite(float(value)) or not 0 < float(value) <= 1:
                raise ValueError(
                    f"Farm '{self.name}': {name} must be greater than 0 and no greater than 1")
        if not self.crop_type:
            raise ValueError(
                f"Farm '{self.name}': 'crop_type' is required in plot settings. "
                f"Set it to the actual crop being grown (e.g. 'maize', 'beans'). "

            )
        if not self.planting_date:
            raise ValueError(
                f"Farm '{self.name}': 'planting_date' is required in plot settings. "
                f"Set it to the actual date the crop was planted (YYYY-MM-DD). "
                f"GDD cannot be computed without an explicit planting date."
            )

        import pandas as pd
        try:
            planting_timestamp = pd.Timestamp(self.planting_date)
        except Exception as exc:
            raise ValueError(
                f"Farm '{self.name}': planting_date '{self.planting_date}' "
                f"is not a valid date. Use ISO format: YYYY-MM-DD "
                f"(e.g. '2024-04-15')."
            ) from exc
        if self.harvest_date:
            try:
                harvest_timestamp = pd.Timestamp(self.harvest_date)
            except Exception as exc:
                raise ValueError(
                    f"Farm '{self.name}': harvest_date "
                    f"'{self.harvest_date}' is not a valid date") from exc
            if harvest_timestamp < planting_timestamp:
                raise ValueError(
                    f"Farm '{self.name}': harvest_date cannot be before "
                    "planting_date")
        if not math.isfinite(float(self.initial_gdd)) or float(
                self.initial_gdd) < 0:
            raise ValueError(
                f"Farm '{self.name}': initial_gdd must be a non-negative "
                "finite value")

        from crops import CROP_PARAMS
        if self.crop_type not in CROP_PARAMS:
            known = list(CROP_PARAMS)
            raise ValueError(
                f"Farm '{self.name}': crop_type '{self.crop_type}' is not "
                f"in the crop database. Known crops: {known}. "
                f"Add the crop to crops.py or correct the plot settings."
            )

        if self.sensor_kind in ("tension", "both"):
            if self.static_threshold_cbar is None or not math.isfinite(
                    float(self.static_threshold_cbar)) or float(self.static_threshold_cbar) <= 0:
                raise ValueError(
                    f"Farm '{self.name}': a positive field-calibrated "
                    "static_threshold_cbar is required for tension sensors")
        if self.threshold_mode not in {"static", "dynamic"}:
            raise ValueError(
                f"Farm '{self.name}': threshold_mode must be 'static' or 'dynamic'")
        if self.threshold_mode == "dynamic":
            hydraulic = (
                self.field_capacity_vwc, self.wilting_point_vwc,
                self.root_depth_m, self.sensor_depth_m,
            )
            has_hydraulic = all(value is not None for value in hydraulic) and bool(
                self.soil_water_retention_curve)
            required_stage_keys = {
                "pre_emergence", "development", "mid_season",
                "late_season", "post_maturity",
            }
            has_stage_curve = required_stage_keys.issubset(
                self.stage_thresholds_cbar)
            has_boundaries = bool(self.soil_water_retention_curve) and all(
                v is not None for v in (self.field_capacity_lower, self.permanent_wilting_point))
            if has_boundaries and not has_hydraulic:
                from soil_calibration import validate_soil_calibration
                validate_soil_calibration(self.soil_water_retention_curve,
                    self.field_capacity_lower, self.permanent_wilting_point,
                    self.saturation, self.soil_calibration)
            if not has_hydraulic and not has_stage_curve and not has_boundaries:
                raise ValueError(
                    f"Farm '{self.name}': dynamic mode requires hydraulic "
                    "VWC/root/sensor/retention-curve inputs or all five "
                    "stage_thresholds_cbar values")
            if has_hydraulic:
                theta_fc, theta_wp, root_depth, sensor_depth = map(
                    float, hydraulic)
                if not (0 < theta_wp < theta_fc < 1):
                    raise ValueError(
                        f"Farm '{self.name}': require 0 < wilting_point_vwc "
                        "< field_capacity_vwc < 1")
                if root_depth <= 0 or sensor_depth <= 0 or sensor_depth > root_depth:
                    raise ValueError(
                        f"Farm '{self.name}': sensor/root depths must be "
                        "positive and sensor depth cannot exceed root depth")
        if self.depletion_fraction is not None and not (
                0 < float(self.depletion_fraction) < 1):
            raise ValueError(
                f"Farm '{self.name}': depletion_fraction must be between 0 and 1")
        required_stage_keys = {
            "pre_emergence", "development", "mid_season",
            "late_season", "post_maturity",
        }
        if self.stage_depletion_fractions:
            if set(self.stage_depletion_fractions) != required_stage_keys:
                raise ValueError(
                    f"Farm '{self.name}': stage_depletion_fractions must "
                    "contain exactly all five crop stages")
            if any(
                not math.isfinite(float(value)) or not 0 < float(value) < 1
                for value in self.stage_depletion_fractions.values()
            ):
                raise ValueError(
                    f"Farm '{self.name}': every stage depletion fraction "
                    "must be between 0 and 1")
        if self.stage_thresholds_cbar:
            if set(self.stage_thresholds_cbar) != required_stage_keys:
                raise ValueError(
                    f"Farm '{self.name}': stage_thresholds_cbar must "
                    "contain exactly all five crop stages")
            if any(
                not math.isfinite(float(value)) or float(value) <= 0
                for value in self.stage_thresholds_cbar.values()
            ):
                raise ValueError(
                    f"Farm '{self.name}': every stage threshold must be "
                    "a positive cbar value")
        if not math.isfinite(float(self.threshold_hysteresis_cbar)) or float(
                self.threshold_hysteresis_cbar) < 0:
            raise ValueError(
                f"Farm '{self.name}': threshold_hysteresis_cbar cannot be negative")
        if self.soil_texture_class is not None:
            if not (0 <= self.soil_texture_class <= 11):
                raise ValueError(
                    f"Farm '{self.name}': soil_texture_class must be 0-11 "
                    f"(USDA texture classes), got {self.soil_texture_class}."
                )
        if self.irrigation_mode in {"automatic", "approval_required"}:
            missing = []
            if not self.flow_sensor_ids:
                missing.append("actuator")
            try:
                area = float(self.plot_area_m2)
            except (TypeError, ValueError):
                area = math.nan
            if not math.isfinite(area) or area <= 0:
                missing.append("positive_plot_area_m2")
            if missing:
                raise ValueError(
                    f"Farm '{self.name}': calculated irrigation configuration "
                    f"is incomplete: {missing}")

    def apply_to_plot(self, plot) -> None:
        """Apply resolved agronomic fields to a legacy Plot runtime object."""
        # This is an in-memory adapter; SQLite owns persisted settings.
        plot.user_given_name = self.name
        plot.gps_info = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "lattitude": self.latitude,
        }
        plot.enable_experimental_ndre_kc = self.enable_experimental_ndre_kc
        plot.timezone = self.timezone
        plot.sensor_kind = self.sensor_kind
        plot.irrigation_type = self.irrigation_type or "unknown"
        plot.irrigation_mode = self.irrigation_mode or ""
        plot.application_efficiency = float(self.application_efficiency)
        plot.effective_rainfall_fraction = float(
            self.effective_rainfall_fraction)
        plot.crop_type = self.crop_type
        plot.planting_date = self.planting_date
        plot.harvest_date = self.harvest_date
        plot.initial_gdd = float(self.initial_gdd)
        plot.soil_texture_class = self.soil_texture_class
        plot.threshold_static = self.static_threshold_cbar
        plot.threshold = self.static_threshold_cbar
        plot.threshold_mode = self.threshold_mode
        for name in (
            "soil_water_retention_curve", "plot_area_m2",
            "field_capacity_vwc", "wilting_point_vwc", "root_depth_m",
            "sensor_depth_m", "depletion_fraction",
            "threshold_hysteresis_cbar", "stage_depletion_fractions",
            "stage_thresholds_cbar",
        ):
            setattr(plot, name, getattr(self, name))
        if self.field_capacity_lower is not None:
            plot.field_capacity_lower = self.field_capacity_lower
        if self.permanent_wilting_point is not None:
            plot.permanent_wilting_point = self.permanent_wilting_point
        if self.field_capacity_lower is not None or self.permanent_wilting_point is not None:
            plot.saturation = self.saturation
            plot.soil_calibration = dict(self.soil_calibration)
        plot.forecast_interval_minutes = self.forecast_interval_minutes
        plot.forecast_horizon_days = self.forecast_horizon_days
        plot.predict_period_hours = self.predict_period_hours
        plot.retrain_interval_days = self.retraining_interval_days
        plot.irrigation_confirmation_seconds = self.irrigation_confirmation_seconds
        plot.error_retry_seconds = self.error_retry_seconds
        plot.look_ahead_time = self.advise_horizon_hours
        plot.watch_horizon_time = self.watch_horizon_hours
        plot.device_and_sensor_ids_moisture = list(self.moisture_sensor_ids)
        plot.device_and_sensor_ids_temp = list(self.temperature_sensor_ids)
        plot.device_and_sensor_ids_flow = list(self.flow_sensor_ids)
        plot.device_and_sensor_ids_flow_confirmation = list(
            self.flow_confirmation_sensor_ids)
        plot.flow_confirmation_mode = self.flow_confirmation_mode
