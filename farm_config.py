import logging
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

    This is the sole source of truth for all farm configurations. All farm-loading
    functions in the codebase must use this class to prevent field drift hazards.
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

    crop_type: str = ""
    planting_date: str = ""
    harvest_date: Optional[str] = None

    soil_texture_class: Optional[int] = None   # USDA 0-11

    initial_gdd: float = 0.0

    advise_horizon_hours: float = 24.0
    watch_horizon_hours:  float = 72.0

    # Runtime-populated sensor column categorization (tension/capacitive/converted only)
    tension_sensor_cols:  List[str] = field(default_factory=list)
    moisture_sensor_cols: List[str] = field(default_factory=list)
    temp_sensor_cols:     List[str] = field(default_factory=list)
    battery_cols:         List[str] = field(default_factory=list)
    resistance_cols:      List[str] = field(default_factory=list)
    other_cols:           List[str] = field(default_factory=list)

    def __post_init__(self):
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


def _parse_entry(farm_id: str, entry: dict) -> FarmConfig:
    """Parse one YAML entry into a FarmConfig with clear missing-field errors."""
    # Check core required fields first
    missing_core = [k for k in _REQUIRED_CORE if k not in entry]
    if missing_core:
        raise ValueError(
            f"Farm '{farm_id}' missing required fields: {missing_core}")

    # Check phenology required fields with clear guidance
    for key in _REQUIRED_PHENO:
        if key not in entry or not entry[key]:
            raise ValueError(
                f"Farm '{farm_id}': '{key}' is required for the phenology pipeline. "
                f"Add it to farms.yaml."
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
        crop_type=entry["crop_type"],
        # coerce int/date YAML values to str
        planting_date=str(entry["planting_date"]),
        harvest_date=str(entry["harvest_date"]) if entry.get(
            "harvest_date") else None,
        soil_texture_class=entry.get("soil_texture_class", None),
        initial_gdd=float(entry.get("initial_gdd", 0.0)),
        advise_horizon_hours=float(entry.get("advise_horizon_hours", 24.0)),
        watch_horizon_hours=float(entry.get("watch_horizon_hours", 72.0)),
    )
