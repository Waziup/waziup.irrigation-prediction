"""

Sources and status:
    FAO-56 supports the Kc method and representative Kc values.
    McMaster & Wilhelm (1997) discusses GDD calculation methods.



Adding a new crop:
    1. Add a CropParams entry to CROP_PARAMS.
    2. Verify GDD thresholds against field trial data for the target region.
    3. Run phenology_engine --list-crops to confirm the entry is picked up.
"""

from dataclasses import dataclass
from typing import Dict, Final


@dataclass(frozen=True)
class CropParams:
    """
    Immutable crop-specific parameters for the phenology engine.

    Kc and depletion-fraction values are FAO-style priors. GDD thresholds are
    engineering estimates and require local validation.
    """
    name: str

    # Temperature bounds for GDD accumulation
    t_base: float       # °C — below this, no development
    t_ceiling: float    # °C — above this, enzymes denature; clip before averaging

    # GDD breakpoints — cumulative GDD at which each stage begins
    gdd_emergence: float    # seed → visible seedling
    gdd_dev_end: float      # canopy expansion complete
    gdd_mid_end: float      # flowering / grain-fill complete
    gdd_maturity: float     # physiological maturity

    # FAO-56 crop coefficients
    kc_ini: float   # initial stage (bare soil + sparse seedlings)
    kc_mid: float   # mid-season peak (full canopy)
    kc_end: float   # late-season end (senescence)

    # FAO-56 Table 22 depletion fraction for ETc near 5 mm/day. Installations
    # may override this with a locally calibrated value.
    depletion_fraction: float = 0.50

    # Perennials (olive, grape) reset GDD at the annual phenological start
    # date rather than at a planting event.
    is_perennial: bool = False

    # Phase-0 validation lag between a soil-tension change and the satellite
    # canopy response that should be visible in the validation signal.
    validation_lag_days: float = 5.0


CROP_PARAMS: Final[Dict[str, CropParams]] = {
    "maize": CropParams(
        name="Maize (Zea mays)",
        t_base=10.0, t_ceiling=30.0,
        gdd_emergence=100, gdd_dev_end=700, gdd_mid_end=1100, gdd_maturity=1400,
        kc_ini=0.30, kc_mid=1.20, kc_end=0.35,
        depletion_fraction=0.55,
        validation_lag_days=5.0,
    ),
    "beans": CropParams(
        name="Common Beans (Phaseolus vulgaris)",
        t_base=10.0, t_ceiling=30.0,
        gdd_emergence=80, gdd_dev_end=400, gdd_mid_end=700, gdd_maturity=1100,
        kc_ini=0.35, kc_mid=1.10, kc_end=0.30,
        depletion_fraction=0.45,
        validation_lag_days=4.0,
    ),
    "wheat": CropParams(
        name="Wheat (Triticum aestivum)",
        t_base=0.0, t_ceiling=26.0,
        gdd_emergence=120, gdd_dev_end=500, gdd_mid_end=1000, gdd_maturity=1500,
        kc_ini=0.30, kc_mid=1.15, kc_end=0.25,
        depletion_fraction=0.55,
        validation_lag_days=6.0,
    ),
    "tomato": CropParams(
        name="Tomato (Solanum lycopersicum)",
        t_base=10.0, t_ceiling=30.0,
        gdd_emergence=90, gdd_dev_end=450, gdd_mid_end=800, gdd_maturity=1200,
        kc_ini=0.45, kc_mid=1.15, kc_end=0.70,
        depletion_fraction=0.40,
        validation_lag_days=3.0,
    ),
    "sorghum": CropParams(
        name="Sorghum (Sorghum bicolor)",
        t_base=10.0, t_ceiling=38.0,    # higher ceiling: heat-tolerant
        gdd_emergence=100, gdd_dev_end=600, gdd_mid_end=900, gdd_maturity=1500,
        kc_ini=0.30, kc_mid=1.00, kc_end=0.55,
        depletion_fraction=0.55,
        validation_lag_days=6.0,
    ),
    "rice": CropParams(
        name="Rice (Oryza sativa)",

        t_base=10.0, t_ceiling=35.0,
        gdd_emergence=100, gdd_dev_end=650, gdd_mid_end=1050, gdd_maturity=1450,
        # FAO-56 rice priors; flooded-soil evaporation is not modelled here.
        kc_ini=1.05, kc_mid=1.20, kc_end=0.90,
        depletion_fraction=0.20,
        validation_lag_days=5.0,
    ),
    "olive": CropParams(
        name="Olive (Olea europaea)",
        t_base=7.0, t_ceiling=35.0,
        gdd_emergence=0, gdd_dev_end=400, gdd_mid_end=1200, gdd_maturity=2000,
        kc_ini=0.55, kc_mid=0.70, kc_end=0.65,
        depletion_fraction=0.65,
        is_perennial=True,
        validation_lag_days=8.0,
    ),

}


# GROWTH STAGE CONSTANTS

STAGE_PRE_EMERGENCE = 0
STAGE_DEVELOPMENT = 1
STAGE_MID_SEASON = 2
STAGE_LATE_SEASON = 3
STAGE_POST_MATURITY = 4

STAGE_NAMES: Final[Dict[int, str]] = {
    STAGE_PRE_EMERGENCE: "Pre-emergence",
    STAGE_DEVELOPMENT:   "Development",
    STAGE_MID_SEASON:    "Mid-season",
    STAGE_LATE_SEASON:   "Late-season",
    STAGE_POST_MATURITY: "Post-maturity",
}

# SOIL TEXTURE -> IRRIGATION TRIGGER BASE LOOKUP
#
# Legacy texture-only trigger estimates.  These are retained for analysis and
# migration diagnostics but are not used by the production threshold path.
# Soil texture alone cannot define a safe tension trigger.
# Keys match TEXTURE_CLASS_MAP in fetch_soil_data.py.

IRRIGATION_TRIGGER_BASE_BY_TEXTURE: Final[Dict[int, float]] = {
    0:  35.0,   # Sand
    1:  37.0,   # Loamy Sand
    2:  40.0,   # Sandy Loam
    3:  45.0,   # Loam (reference default)
    4:  42.0,   # Silt Loam
    5:  40.0,   # Silt
    6:  43.0,   # Sandy Clay Loam
    7:  50.0,   # Clay Loam
    8:  52.0,   # Silty Clay Loam
    9:  48.0,   # Sandy Clay
    10: 55.0,   # Silty Clay
    11: 55.0,   # Clay
}

# Loam — safe fallback when texture class is unknown
IRRIGATION_TRIGGER_BASE_DEFAULT = 45.0


def get_crop_params(crop_type: str) -> CropParams:

    if crop_type not in CROP_PARAMS:
        known = list(CROP_PARAMS)
        raise KeyError(
            f"Unknown crop_type '{crop_type}'. "
            f"Known crops: {known}. Add the crop to crops.py."
        )
    return CROP_PARAMS[crop_type]
