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
    irrigation_type: str = ""
    irrigation_mode: str = ""
    # Device identifiers may originate in the compatibility JSON layer, but
    # YAML values override them when explicitly configured.
    moisture_sensor_ids: List[str] = field(default_factory=list)
    temperature_sensor_ids: List[str] = field(default_factory=list)
    flow_sensor_ids: List[str] = field(default_factory=list)
    flow_confirmation_sensor_ids: List[str] = field(default_factory=list)

    crop_type: str = ""
    planting_date: str = ""
    harvest_date: Optional[str] = None

    soil_texture_class: Optional[int] = None   # USDA 0-11

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
        if self.retraining_interval_days <= 0 or self.irrigation_confirmation_seconds <= 0 or self.error_retry_seconds <= 0:
            raise ValueError(
                f"Farm '{self.farm_id}': retraining, confirmation, and retry intervals must be positive"
            )
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
            pd.Timestamp(self.planting_date)
        except Exception as exc:
            raise ValueError(
                f"Farm '{self.farm_id}': planting_date '{self.planting_date}' "
                f"is not a valid date. Use ISO format: YYYY-MM-DD "
                f"(e.g. '2024-04-15')."
            ) from exc

        from crops import CROP_PARAMS
        if self.crop_type not in CROP_PARAMS:
            known = [k for k in CROP_PARAMS if k != "generic"]
            raise ValueError(
                f"Farm '{self.farm_id}': crop_type '{self.crop_type}' is not "
                f"in the crop database. Known crops: {known}. "
                f"Add the crop to crops.py or correct the farms.yaml entry."
            )
        if self.crop_type == "generic":
            raise ValueError(
                f"Farm '{self.farm_id}': crop_type 'generic' is not a valid "
                f"production crop. Set the actual crop being grown. "
                f"Known crops: {[k for k in CROP_PARAMS if k != 'generic']}."
            )

        if self.soil_texture_class is not None:
            if not (0 <= self.soil_texture_class <= 11):
                raise ValueError(
                    f"Farm '{self.farm_id}': soil_texture_class must be 0-11 "
                    f"(USDA texture classes), got {self.soil_texture_class}."
                )
        elif self.sensor_kind in ("tension", "both"):
            raise ValueError(
                f"Farm '{self.farm_id}': soil_texture_class is required for "
                f"tension sensors. Set the USDA texture class in farms.yaml "
                f"so irrigation thresholds do not fall back to the wrong "
                f"Loam default."
            )
        else:
            log.warning(
                "Farm '%s': soil_texture_class not set — using default trigger base only for non-tension sensors.",
                self.farm_id,
            )

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
        plot.timezone = self.timezone
        plot.sensor_kind = self.sensor_kind
        plot.irrigation_type = self.irrigation_type or "unknown"
        plot.irrigation_mode = self.irrigation_mode or ""
        plot.crop_type = self.crop_type
        plot.planting_date = self.planting_date
        plot.soil_texture_class = self.soil_texture_class
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
            "soil_texture_class") is None:
        missing_agronomic.append("soil_texture_class")
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
        sensor_kind=entry["sensor_kind"],
        # normalise None/absent → ""
        irrigation_type=entry.get("irrigation_type") or "",
        irrigation_mode=entry.get("irrigation_mode") or "",
        moisture_sensor_ids=list(entry.get("moisture_sensor_ids", [])),
        temperature_sensor_ids=list(entry.get("temperature_sensor_ids", [])),
        flow_sensor_ids=list(entry.get("flow_sensor_ids", [])),
        flow_confirmation_sensor_ids=list(entry.get(
            "flow_confirmation_sensor_ids", [])),
        crop_type=entry["crop_type"],
        # coerce int/date YAML values to str
        planting_date=str(entry["planting_date"]),
        harvest_date=str(entry["harvest_date"]) if entry.get(
            "harvest_date") else None,
        soil_texture_class=entry.get("soil_texture_class", None),
        initial_gdd=float(entry.get("initial_gdd", 0.0)),
        advise_horizon_hours=float(entry.get("advise_horizon_hours", 24.0)),
        watch_horizon_hours=float(entry.get("watch_horizon_hours", 72.0)),
        forecast_interval_minutes=int(
            entry.get("forecast_interval_minutes", 60)),
        forecast_horizon_days=float(entry.get("forecast_horizon_days", 5.0)),
        predict_period_hours=float(entry.get("predict_period_hours", 3.0)),
        retraining_interval_days=float(
            entry.get("retraining_interval_days", 1.0)),
        irrigation_confirmation_seconds=int(
            entry.get("irrigation_confirmation_seconds", 10800)),
        error_retry_seconds=int(entry.get("error_retry_seconds", 1800)),
    )
