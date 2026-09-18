import unittest
from datetime import date

import numpy as np
import pandas as pd

import phenology_engine as pe
import crops


class TestPhenologyIntegration(unittest.TestCase):
    def test_growth_stage_transitions_follow_gdd_breakpoints(self):
        maize = crops.get_crop_params("maize")
        self.assertEqual(pe.get_growth_stage(0.0, "maize"), crops.STAGE_PRE_EMERGENCE)
        self.assertEqual(pe.get_growth_stage(maize.gdd_emergence - 1, "maize"), crops.STAGE_PRE_EMERGENCE)
        self.assertEqual(pe.get_growth_stage(maize.gdd_emergence, "maize"), crops.STAGE_DEVELOPMENT)
        self.assertEqual(pe.get_growth_stage(maize.gdd_dev_end, "maize"), crops.STAGE_MID_SEASON)
        self.assertEqual(pe.get_growth_stage(maize.gdd_mid_end, "maize"), crops.STAGE_LATE_SEASON)
        self.assertEqual(pe.get_growth_stage(maize.gdd_maturity, "maize"), crops.STAGE_POST_MATURITY)

    def test_dynamic_threshold_changes_by_stage(self):
        base = 45.0
        self.assertAlmostEqual(pe.get_dynamic_threshold(0.0, "maize", base), 45.0)
        self.assertAlmostEqual(pe.get_dynamic_threshold(200.0, "maize", base), 40.0)
        self.assertAlmostEqual(pe.get_dynamic_threshold(800.0, "maize", base), 30.0)
        self.assertAlmostEqual(pe.get_dynamic_threshold(1200.0, "maize", base), 55.0)

    def test_gdd_series_and_kc_curve_are_monotonic_in_expected_ranges(self):
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        temp_max = pd.Series([28.0] * len(idx), index=idx)
        temp_min = pd.Series([12.0] * len(idx), index=idx)
        gdd = pe.compute_gdd_series(temp_max, temp_min, "2024-01-01", "maize")
        self.assertTrue((gdd.diff().fillna(0.0) >= 0).all())

        kc = pe.compute_kc_gdd_series(gdd, "maize")
        self.assertGreaterEqual(float(kc.iloc[0]), 0.3)
        self.assertLessEqual(float(kc.iloc[-1]), 1.2)

    def test_date_window_contract_for_weather_and_satellite_requests(self):
        with self.assertRaises(ValueError):
            import spaceiotbox_client as client
            client.fetch_weather_frame(0.0, 0.0, start_date="2024-02-10", end_date="2024-02-01")

        days_since_planting = (date.today() - date(2024, 1, 5)).days
        expected = max(days_since_planting + 14, 30)
        self.assertGreaterEqual(expected, 30)
        self.assertNotEqual(expected, 5)

    def test_satellite_validation_is_neutral_when_data_is_sparse(self):
        resp = pe.check_satellite_tension_consistency(
            crop_type="maize",
            growth_stage=pe.STAGE_DEVELOPMENT,
            gdd_cumulative=150.0,
            current_tension=32.0,
            stress_threshold_cbar=45.0,
            tension_forecast={"24h": 35.0, "48h": 40.0, "72h": 44.0},
            satellite_history=None,
            satellite_ndvi=np.nan,
            satellite_ndre=np.nan,
            satellite_ndvi_age_hours=float("inf"),
            satellite_data_age_hours=float("inf"),
        )
        self.assertTrue(resp["insufficient_data"])
        self.assertEqual(resp["direction_agreement"], 0.5)


if __name__ == "__main__":
    unittest.main()
