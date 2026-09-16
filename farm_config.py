import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FARMS_PATH = BASE_DIR / "config" / "farms.yaml"

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    log.warning(
        "PyYAML not installed — load_farms() will not work. pip install pyyaml")


_REQUIRED_CORE = ["name", "csv_path", "latitude", "longitude", "timezone",
                  "sensor_kind", "site_id"]

_REQUIRED_PHENO = ["crop_type", "planting_date"]


@dataclass
class FarmConfig:
    """
    Configuration for a single farm or plot.

    crop_type and planting_date are REQUIRED. Absence raises ValueError at
    construction time so failures are immediate and unambiguous.

    NOT frozen: sensor column categorization (tension_sensor_cols, etc.) is
    populated at runtime during CSV parsing and must be mutable. The core config
    (crop_type, planting_date, etc.) is still validated immutably in __post_init__.

    This is the source of truth for farms intentionally configured in YAML.
    Installation/UI-managed plots retain the values persisted in their JSON
    configuration instead of being overwritten during startup.
    """
    farm_id: str
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
            raise ValueError(f"Farm '{self.farm_id}': timezone is required")
        if self.sensor_kind not in {"tension", "capacitive", "both"}:
            raise ValueError(
                f"Farm '{self.farm_id}': unsupported sensor_kind '{self.sensor_kind}'"
            )
        if self.irrigation_mode and self.irrigation_mode not in {
                "automatic", "approval_required", "manual", "advisory_only"}:
            raise ValueError(
                f"Farm '{self.farm_id}': unsupported irrigation_mode "
                f"'{self.irrigation_mode}'"
            )
        if self.flow_confirmation_mode not in {"event", "cumulative"}:
            raise ValueError(
                f"Farm '{self.farm_id}': flow_confirmation_mode must be "
                "'event' or 'cumulative'")
        if not math.isfinite(self.latitude) or not -90 <= self.latitude <= 90:
            raise ValueError(
                f"Farm '{self.farm_id}': latitude must be between -90 and 90")
        if not math.isfinite(self.longitude) or not -180 <= self.longitude <= 180:
            raise ValueError(
                f"Farm '{self.farm_id}': longitude must be between -180 and 180")
        if self.advise_horizon_hours < 0 or self.watch_horizon_hours < 0:
            raise ValueError(
                f"Farm '{self.farm_id}': decision horizons cannot be negative"
            )
        if self.forecast_interval_minutes <= 0:
            raise ValueError(
                f"Farm '{self.farm_id}': forecast_interval_minutes must be positive"
            )
        if self.forecast_horizon_days <= 0:
            raise ValueError(
                f"Farm '{self.farm_id}': forecast_horizon_days must be positive"
            )
        if self.predict_period_hours <= 0:
            raise ValueError(
                f"Farm '{self.farm_id}': predict_period_hours must be positive"
            )
        if (self.retraining_interval_days <= 0
                or self.irrigation_confirmation_seconds <= 0
                or self.error_retry_seconds <= 0):
            raise ValueError(
                f"Farm '{self.farm_id}': retraining and retry intervals must be positive"
            )
        for name, value in (
                ("application_efficiency", self.application_efficiency),
                ("effective_rainfall_fraction", self.effective_rainfall_fraction)):
            if not math.isfinite(float(value)) or not 0 < float(value) <= 1:
                raise ValueError(
                    f"Farm '{self.farm_id}': {name} must be greater than 0 and no greater than 1")
        if not self.crop_type:
            raise ValueError(
                f"Farm '{self.farm_id}': 'crop_type' is required in farms.yaml. "
                f"Set it to the actual crop being grown (e.g. 'maize', 'beans'). "

            )
        if not self.planting_date:
            raise ValueError(
                f"Farm '{self.farm_id}': 'planting_date' is required in farms.yaml. "
                f"Set it to the actual date the crop was planted (YYYY-MM-DD). "
                f"GDD cannot be computed without an explicit planting date."
            )

        import pandas as pd
        try:
            planting_timestamp = pd.Timestamp(self.planting_date)
        except Exception as exc:
            raise ValueError(
                f"Farm '{self.farm_id}': planting_date '{self.planting_date}' "
                f"is not a valid date. Use ISO format: YYYY-MM-DD "
                f"(e.g. '2024-04-15')."
            ) from exc
        if self.harvest_date:
            try:
                harvest_timestamp = pd.Timestamp(self.harvest_date)
            except Exception as exc:
                raise ValueError(
                    f"Farm '{self.farm_id}': harvest_date "
                    f"'{self.harvest_date}' is not a valid date") from exc
            if harvest_timestamp < planting_timestamp:
                raise ValueError(
                    f"Farm '{self.farm_id}': harvest_date cannot be before "
                    "planting_date")
        if not math.isfinite(float(self.initial_gdd)) or float(
                self.initial_gdd) < 0:
            raise ValueError(
                f"Farm '{self.farm_id}': initial_gdd must be a non-negative "
                "finite value")

        from crops import CROP_PARAMS
        if self.crop_type not in CROP_PARAMS:
            known = list(CROP_PARAMS)
            raise ValueError(
                f"Farm '{self.farm_id}': crop_type '{self.crop_type}' is not "
                f"in the crop database. Known crops: {known}. "
                f"Add the crop to crops.py or correct the farms.yaml entry."
            )

        if self.sensor_kind in ("tension", "both"):
            if self.static_threshold_cbar is None or not math.isfinite(
                    float(self.static_threshold_cbar)) or float(self.static_threshold_cbar) <= 0:
                raise ValueError(
                    f"Farm '{self.farm_id}': a positive field-calibrated "
                    "static_threshold_cbar is required for tension sensors")
        if self.threshold_mode not in {"static", "dynamic"}:
            raise ValueError(
                f"Farm '{self.farm_id}': threshold_mode must be 'static' or 'dynamic'")
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
                    f"Farm '{self.farm_id}': dynamic mode requires hydraulic "
                    "VWC/root/sensor/retention-curve inputs or all five "
                    "stage_thresholds_cbar values")
            if has_hydraulic:
                theta_fc, theta_wp, root_depth, sensor_depth = map(
                    float, hydraulic)
                if not (0 < theta_wp < theta_fc < 1):
                    raise ValueError(
                        f"Farm '{self.farm_id}': require 0 < wilting_point_vwc "
                        "< field_capacity_vwc < 1")
                if root_depth <= 0 or sensor_depth <= 0 or sensor_depth > root_depth:
                    raise ValueError(
                        f"Farm '{self.farm_id}': sensor/root depths must be "
                        "positive and sensor depth cannot exceed root depth")
        if self.depletion_fraction is not None and not (
                0 < float(self.depletion_fraction) < 1):
            raise ValueError(
                f"Farm '{self.farm_id}': depletion_fraction must be between 0 and 1")
        required_stage_keys = {
            "pre_emergence", "development", "mid_season",
            "late_season", "post_maturity",
        }
        if self.stage_depletion_fractions:
            if set(self.stage_depletion_fractions) != required_stage_keys:
                raise ValueError(
                    f"Farm '{self.farm_id}': stage_depletion_fractions must "
                    "contain exactly all five crop stages")
            if any(
                not math.isfinite(float(value)) or not 0 < float(value) < 1
                for value in self.stage_depletion_fractions.values()
            ):
                raise ValueError(
                    f"Farm '{self.farm_id}': every stage depletion fraction "
                    "must be between 0 and 1")
        if self.stage_thresholds_cbar:
            if set(self.stage_thresholds_cbar) != required_stage_keys:
                raise ValueError(
                    f"Farm '{self.farm_id}': stage_thresholds_cbar must "
                    "contain exactly all five crop stages")
            if any(
                not math.isfinite(float(value)) or float(value) <= 0
                for value in self.stage_thresholds_cbar.values()
            ):
                raise ValueError(
                    f"Farm '{self.farm_id}': every stage threshold must be "
                    "a positive cbar value")
        if not math.isfinite(float(self.threshold_hysteresis_cbar)) or float(
                self.threshold_hysteresis_cbar) < 0:
            raise ValueError(
                f"Farm '{self.farm_id}': threshold_hysteresis_cbar cannot be negative")
        if self.soil_texture_class is not None:
            if not (0 <= self.soil_texture_class <= 11):
                raise ValueError(
                    f"Farm '{self.farm_id}': soil_texture_class must be 0-11 "
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
                    f"Farm '{self.farm_id}': calculated irrigation configuration "
                    f"is incomplete: {missing}")

    def apply_to_plot(self, plot) -> None:
        """Apply authoritative farm fields to a legacy Plot runtime object."""
        # The existing Plot object remains the runtime carrier while YAML owns
        # agronomic, location, sensor, decision, and timing values.
        plot.farm_id = self.farm_id
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


def load_farm_config(
    farm_id: str,
    yaml_path: str = str(DEFAULT_FARMS_PATH),
) -> FarmConfig:
    """
    Load a single farm's configuration from farms.yaml.

    Raises:
        FileNotFoundError  — yaml_path does not exist
        KeyError           — farm_id not found in file
        ValueError         — required fields missing or invalid
    """
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required: pip install pyyaml")
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"farms.yaml not found: {path.resolve()}")
    with open(path, encoding="utf-8") as f:
        all_farms = yaml.safe_load(f) or {}
    if not isinstance(all_farms, dict):
        raise ValueError(
            f"Invalid farms.yaml root in {yaml_path}: expected mapping/dict, got {type(all_farms).__name__}"
        )
    if farm_id not in all_farms:
        raise KeyError(
            f"Farm '{farm_id}' not in {yaml_path}. "
            f"Available: {list(all_farms)}"
        )
    return _parse_entry(farm_id, all_farms[farm_id])


def load_all_farms(
    yaml_path: str = str(DEFAULT_FARMS_PATH),
    strict: bool = True,
) -> Dict[str, FarmConfig]:
    """
    Load all farm configurations from farms.yaml.

    By default this fails fast when any farm entry is invalid, which prevents
    partial deployments where only a subset of farms are loaded silently.

    Set strict=False to retain best-effort behavior (invalid farms are logged
    and skipped) for ad-hoc debugging workflows.
    """
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required: pip install pyyaml")
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"farms.yaml not found: {path.resolve()}")
    with open(path, encoding="utf-8") as f:
        all_farms = yaml.safe_load(f) or {}
    if not isinstance(all_farms, dict):
        raise ValueError(
            f"Invalid farms.yaml root in {yaml_path}: expected mapping/dict, got {type(all_farms).__name__}"
        )
    configs: Dict[str, FarmConfig] = {}
    invalid_entries = []
    for fid, entry in all_farms.items():
        if isinstance(entry, dict) and entry.get("enabled", True) is False:
            log.info("Farm '%s' is disabled; skipping validation and runtime loading", fid)
            continue
        try:
            configs[fid] = _parse_entry(fid, entry)
        except (KeyError, ValueError) as e:
            if strict:
                invalid_entries.append((fid, str(e)))
            else:
                log.error("Skipping farm '%s': %s", fid, e)

    if strict and invalid_entries:
        problems = "; ".join(
            f"{fid}: {reason}" for fid, reason in invalid_entries
        )
        raise ValueError(
            f"Invalid farm configuration(s) in {yaml_path}: {problems}"
        )

    log.info(
        "Loaded %s/%s farm configs from %s",
        len(configs),
        len(all_farms),
        yaml_path,
    )
    return configs


def apply_farm_configs_to_plots(
    plots,
    configs: Dict[str, FarmConfig],
    preserve_ui_config: bool = False,
) -> None:
    """Apply manual YAML configuration without overwriting UI-owned plots.

    ``farms.yaml`` remains authoritative for plots deliberately managed there.
    A plot saved by the installation/settings UI carries
    ``configuration_source == 'ui'`` and is skipped when
    ``preserve_ui_config`` is enabled.
    """
    plot_values = list(plots.values()) if isinstance(
        plots, dict) else list(plots)
    if not preserve_ui_config and len(plot_values) != len(configs):
        raise ValueError(
            f"Farm/plot count mismatch: {len(configs)} farms in farms.yaml, "
            f"{len(plot_values)} runtime plots loaded"
        )

    ordered_configs = list(configs.values())
    used_ids = set()
    for position, plot in enumerate(plot_values):
        if preserve_ui_config and getattr(
                plot, "configuration_source", "legacy") == "ui":
            continue
        farm_id = getattr(plot, "farm_id", None)
        farm = configs.get(farm_id) if farm_id else None
        if farm is None:
            # Older JSON files have no farm_id, so preserve deterministic order
            # during migration instead of guessing from names or coordinates.
            farm = ordered_configs[position] if position < len(
                ordered_configs) else None
        if farm is None:
            log.info(
                "No valid manual YAML farm for runtime plot %s; keeping installation configuration",
                getattr(plot, "id", position + 1),
            )
            continue
        if farm.farm_id in used_ids:
            raise ValueError(
                f"Farm '{farm.farm_id}' is assigned to more than one plot")
        used_ids.add(farm.farm_id)
        farm.apply_to_plot(plot)


def _parse_entry(farm_id: str, entry: dict) -> FarmConfig:
    """Parse one YAML entry into a FarmConfig with clear missing-field errors."""
    # Check core required fields first
    missing_core = [k for k in _REQUIRED_CORE if k not in entry]
    if missing_core:
        raise ValueError(
            f"Farm '{farm_id}' missing required fields: {missing_core}")

    # Report the complete agronomic gap in one startup attempt.
    missing_agronomic = [
        key for key in _REQUIRED_PHENO if key not in entry or not entry[key]
    ]
    if entry.get("sensor_kind") in {"tension", "both"} and entry.get(
            "static_threshold_cbar") is None:
        missing_agronomic.append("static_threshold_cbar")
    if missing_agronomic:
        raise ValueError(
            f"Farm '{farm_id}' is missing agronomic fields required by the "
            f"phenology pipeline: {missing_agronomic}. Add them to farms.yaml."
        )

    return FarmConfig(
        farm_id=farm_id,
        name=entry["name"],
        site_id=int(entry["site_id"]),
        csv_path=entry["csv_path"],
        latitude=float(entry["latitude"]),
        longitude=float(entry["longitude"]),
        timezone=entry["timezone"],
        enable_experimental_ndre_kc=bool(
            entry.get("enable_experimental_ndre_kc", False)),
        sensor_kind=entry["sensor_kind"],
        # normalise None/absent → ""
        irrigation_type=entry.get("irrigation_type") or "",
        irrigation_mode=entry.get("irrigation_mode") or "",
        application_efficiency=float(entry.get("application_efficiency", 0.85)),
        effective_rainfall_fraction=float(entry.get(
            "effective_rainfall_fraction", 0.80)),
        moisture_sensor_ids=list(entry.get("moisture_sensor_ids", [])),
        temperature_sensor_ids=list(entry.get("temperature_sensor_ids", [])),
        flow_sensor_ids=list(entry.get("flow_sensor_ids", [])),
        flow_confirmation_sensor_ids=list(entry.get(
            "flow_confirmation_sensor_ids", [])),
        flow_confirmation_mode=str(entry.get(
            "flow_confirmation_mode", "event")).strip().lower(),
        crop_type=entry["crop_type"],
        # coerce int/date YAML values to str
        planting_date=str(entry["planting_date"]),
        harvest_date=str(entry["harvest_date"]) if entry.get(
            "harvest_date") else None,
        soil_texture_class=entry.get("soil_texture_class", None),
        static_threshold_cbar=entry.get("static_threshold_cbar"),
        threshold_mode=str(entry.get(
            "threshold_mode",
            "dynamic" if entry.get("use_dynamic_threshold", False) else "static",
        )).strip().lower(),
        field_capacity_lower=entry.get("field_capacity_lower"),
        permanent_wilting_point=entry.get("permanent_wilting_point"),
        saturation=float(entry.get("saturation", 0)),
        soil_calibration=dict(entry.get("soil_calibration", {}) or {}),
        field_capacity_vwc=entry.get("field_capacity_vwc"),
        wilting_point_vwc=entry.get("wilting_point_vwc"),
        root_depth_m=entry.get("root_depth_m"),
        sensor_depth_m=entry.get("sensor_depth_m"),
        depletion_fraction=entry.get("depletion_fraction"),
        threshold_hysteresis_cbar=float(entry.get(
            "threshold_hysteresis_cbar", 0.0)),
        stage_depletion_fractions=dict(entry.get(
            "stage_depletion_fractions", {}) or {}),
        stage_thresholds_cbar=dict(entry.get(
            "stage_thresholds_cbar", {}) or {}),
        soil_water_retention_curve=list(
            entry.get("soil_water_retention_curve", []) or []),
        plot_area_m2=entry.get("plot_area_m2"),
        initial_gdd=float(entry.get("initial_gdd", 0.0)),
        advise_horizon_hours=float(entry.get("advise_horizon_hours", 24.0)),
        watch_horizon_hours=float(entry.get("watch_horizon_hours", 72.0)),
        forecast_interval_minutes=int(
            entry.get("forecast_interval_minutes", 60)),
        forecast_horizon_days=float(entry.get("forecast_horizon_days", 5.0)),
        predict_period_hours=float(entry.get("predict_period_hours", 3.0)),
        retraining_interval_days=float(
            entry.get("retraining_interval_days", 1.0)),
        irrigation_confirmation_seconds=int(entry.get(
            "irrigation_confirmation_seconds", 10800)),
        error_retry_seconds=int(entry.get("error_retry_seconds", 1800)),
    )
