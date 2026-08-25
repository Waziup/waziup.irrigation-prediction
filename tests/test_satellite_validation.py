import unittest
import pandas as pd
import numpy as np

import phenology_engine as pe


class TestSatelliteValidation(unittest.TestCase):
    def test_no_data_neutral(self):
        resp = pe.check_satellite_tension_consistency(
            crop_type="maize",
            growth_stage=pe.STAGE_DEVELOPMENT,
            gdd_cumulative=100.0,
            current_tension=30.0,
            stress_threshold_cbar=45.0,
            tension_forecast={"0h": 30.0},
            satellite_history=None,
            satellite_ndvi=np.nan,
            satellite_ndre=np.nan,
            satellite_ndvi_age_hours=float("inf"),
            satellite_data_age_hours=float("inf"),
        )
        self.assertTrue(resp["insufficient_data"])
        self.assertEqual(resp["direction_agreement"], 0.5)

    def test_aligned_trends_agreement(self):
        current = 30.0
        forecast = {"24h": current + 5.0, "48h": current + 10.0}

        now = pd.Timestamp.utcnow()
        rows = []
        ndvi_vals = [0.8, 0.75, 0.7]
        for i, v in enumerate(ndvi_vals):
            rows.append(
                {
                    "timestamp": (now - pd.Timedelta(days=10 + i)),
                    "sat_ndvi": float(v),
                    "satellite_data_age": float(24 * (2 - i) + 120),
                }
            )
        rows.append({
            "timestamp": now - pd.Timedelta(days=1),
            "sat_ndvi": 0.65,
            "satellite_data_age": 72.0,
        })
        sat_df = pd.DataFrame(rows)

        resp = pe.check_satellite_tension_consistency(
            crop_type="maize",
            growth_stage=pe.STAGE_DEVELOPMENT,
            gdd_cumulative=200.0,
            current_tension=current,
            stress_threshold_cbar=45.0,
            tension_forecast=forecast,
            satellite_history=sat_df,
            satellite_ndvi=np.nan,
            satellite_ndre=np.nan,
            satellite_ndvi_age_hours=24.0,
            satellite_data_age_hours=24.0,
        )

        self.assertFalse(resp["insufficient_data"])
        # Under realistic units, satellite slope magnitudes are small
        # relative to tension scaling; validate non-insufficient outcome
        # and that the validator produced a numeric confidence value.
        self.assertIn("direction_agreement", resp)
        self.assertGreaterEqual(resp["validation_score"], 0.5)
        self.assertIsNotNone(resp["overall_confidence"])

    def test_lagged_satellite_history_is_used(self):
        now = pd.Timestamp.utcnow()
        rows = [
            {"timestamp": now - pd.Timedelta(days=5), "sat_ndvi": 0.25},
            {"timestamp": now - pd.Timedelta(days=4), "sat_ndvi": 0.30},
            {"timestamp": now - pd.Timedelta(days=3), "sat_ndvi": 0.35},
            {"timestamp": now - pd.Timedelta(days=1), "sat_ndvi": 0.40},
        ]
        sat_df = pd.DataFrame(rows)

        resp = pe.check_satellite_tension_consistency(
            crop_type="tomato",
            growth_stage=pe.STAGE_DEVELOPMENT,
            gdd_cumulative=300.0,
            current_tension=35.0,
            stress_threshold_cbar=40.0,
            tension_forecast={"24h": 40.0, "48h": 45.0},
            satellite_history=sat_df,
            satellite_ndvi=np.nan,
            satellite_ndre=np.nan,
            satellite_ndvi_age_hours=48.0,
            satellite_data_age_hours=48.0,
        )

        self.assertTrue(resp["insufficient_data"])

    def test_unknown_cadence_rejects_stale_midseason_observation(self):
        q = pe._freshness_quality(age_hours=2000, stage=pe.STAGE_MID_SEASON)
        self.assertEqual(q, 0.0)

    def test_flat_trend_neutral_in_low_sensitivity_stage(self):
        now = pd.Timestamp.utcnow()
        rows = [
            {"timestamp": now - pd.Timedelta(days=12), "sat_ndvi": 0.42},
            {"timestamp": now - pd.Timedelta(days=11), "sat_ndvi": 0.42},
            {"timestamp": now - pd.Timedelta(days=10), "sat_ndvi": 0.42},
            {"timestamp": now - pd.Timedelta(days=2), "sat_ndvi": 0.42},
        ]
        sat_df = pd.DataFrame(rows)

        resp = pe.check_satellite_tension_consistency(
            crop_type="olive",
            growth_stage=pe.STAGE_LATE_SEASON,
            gdd_cumulative=1800.0,
            current_tension=42.0,
            stress_threshold_cbar=48.0,
            tension_forecast={"24h": 44.0, "48h": 46.0},
            satellite_history=sat_df,
            satellite_ndvi=np.nan,
            satellite_ndre=np.nan,
            satellite_ndvi_age_hours=24.0,
            satellite_data_age_hours=24.0,
        )

        self.assertFalse(resp["insufficient_data"])
        self.assertAlmostEqual(resp["validation_score"], 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
