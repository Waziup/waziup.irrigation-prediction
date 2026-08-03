import os
import json
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv

load_dotenv('.env')

try:
    from spaceiotbox_client import fetch_weather_frame
    from spaceiotbox_satellite import fetch_satellite_snapshot
    from actuation import get_irrigation_recommendation
except Exception as e:
    # If imports fail, tests will be skipped with a clear message
    fetch_weather_frame = None
    fetch_satellite_snapshot = None
    get_irrigation_recommendation = None
    _import_error = e


@unittest.skipUnless(os.getenv('SPACEIOTBOX_API_KEY'), 'SPACEIOTBOX_API_KEY not set in .env')
class TestKisumuLive(unittest.TestCase):
    def test_fetch_live_kisumu_maize(self):
        if fetch_weather_frame is None or fetch_satellite_snapshot is None or get_irrigation_recommendation is None:
            self.skipTest(f'Missing imports: {_import_error}')

        # Kisumu, Kenya (approx)
        lat = -0.0917
        lon = 34.7680
        planting_date = '2026-01-05'

        # Fetch weather for the period 2026-01-05 .. 2026-05-30
        start_date = planting_date
        end_date = '2026-05-30'

        weather_frame = fetch_weather_frame(lat, lon, start_date, end_date)

        # Fetch satellite snapshot (one-year lookback to capture scenes)
        satellite_snapshot = fetch_satellite_snapshot(
            lat, lon, as_of=end_date, lookback_days=365)

        # Build a minimal plot object compatible with get_irrigation_recommendation
        plot = SimpleNamespace(
            gps_info={'latitude': lat, 'longitude': lon},
            crop_type='maize',
            planting_date=planting_date,
            threshold=25.0,
            soil_texture_class='loam',
            irrigation_type='furrow',
            plot_area_m2=1000.0,
            initial_gdd=0.0,
        )

        recommendation = get_irrigation_recommendation(plot)

        # Diagnostics / Audit Section

        weather_audit = {
            "requested_start": start_date,
            "requested_end": end_date,
            "returned_start": None,
            "returned_end": None,
            "rows": 0,
            "columns": [],
            "missing_by_column": {},
            "coverage_by_column": {},
            "date_range_matches_request": False,
        }

        try:
            weather_audit["returned_start"] = str(weather_frame.index.min())
            weather_audit["returned_end"] = str(weather_frame.index.max())
            weather_audit["rows"] = int(len(weather_frame))
            weather_audit["columns"] = list(weather_frame.columns)

            returned_start_date = str(weather_frame.index.min())[:10]
            returned_end_date = str(weather_frame.index.max())[:10]

            weather_audit["date_range_matches_request"] = (
                returned_start_date == start_date
                and returned_end_date == end_date
            )

            for col in weather_frame.columns:
                missing = int(weather_frame[col].isna().sum())
                total = int(len(weather_frame))

                weather_audit["missing_by_column"][col] = missing

                weather_audit["coverage_by_column"][col] = round(
                    ((total - missing) / total) * 100,
                    2
                ) if total else 0

        except Exception as e:
            weather_audit["error"] = str(e)

        growth_stage_audit = {
            "planting_date": planting_date,
            "analysis_end_date": end_date,
            "days_since_planting": None,
            "gdd_cumulative": recommendation.get("gdd_cumulative"),
            "growth_stage": recommendation.get("growth_stage"),
            "sat_ndvi": recommendation.get("sat_ndvi"),
            "warnings": [],
        }

        try:
            growth_stage_audit["days_since_planting"] = (
                date.fromisoformat(end_date)
                - date.fromisoformat(planting_date)
            ).days

            ndvi = recommendation.get("sat_ndvi")
            stage = recommendation.get("growth_stage")

            if ndvi is not None and ndvi > 0.30 and stage == "Pre-emergence":
                growth_stage_audit["warnings"].append(
                    "NDVI indicates vegetation present but growth stage is Pre-emergence."
                )

            if (
                growth_stage_audit["days_since_planting"] > 30
                and stage == "Pre-emergence"
            ):
                growth_stage_audit["warnings"].append(
                    "Crop is older than 30 days but still classified as Pre-emergence."
                )

            if (
                recommendation.get("gdd_cumulative") is not None
                and growth_stage_audit["days_since_planting"] > 90
                and recommendation.get("gdd_cumulative", 0) < 100
            ):
                growth_stage_audit["warnings"].append(
                    "Very low cumulative GDD for crop age. Weather history may be missing."
                )

        except Exception as e:
            growth_stage_audit["error"] = str(e)

        # Also collect STAC/collection info to help surface scene availability
        try:
            import spaceiotbox_satellite as sbox_sat
            collections = []
            try:
                collections = sbox_sat._discover_collection_names()
            except Exception:
                collections = []

            stac_items = []
            try:
                # search items for the requested window
                from pandas import Timestamp
                stac_items = sbox_sat._search_items(lat, lon, Timestamp(
                    start_date + 'T00:00:00Z'), Timestamp(end_date + 'T23:59:59Z'))
            except Exception:
                stac_items = []
        except Exception:
            collections = []
            stac_items = []

        # Save outputs to data/ for inspection and commit
        out_dir = Path('data') / 'test_kisumu_live_outputs'
        out_dir.mkdir(parents=True, exist_ok=True)

        weather_csv = out_dir / 'weather_frame.csv'
        satellite_csv = out_dir / 'satellite_snapshot.csv'
        recommendation_json = out_dir / 'recommendation.json'
        weather_audit_json = out_dir / 'weather_audit.json'
        growth_stage_audit_json = out_dir / 'growth_stage_audit.json'

        try:
            weather_frame.to_csv(weather_csv)
        except Exception:
            with open(weather_csv, 'w') as fh:
                fh.write(repr(weather_frame))

        try:
            satellite_snapshot.to_csv(satellite_csv)
        except Exception:
            with open(satellite_csv, 'w') as fh:
                fh.write(repr(satellite_snapshot))

        try:
            with open(recommendation_json, 'w') as fh:
                json.dump(recommendation, fh, default=str, indent=2)
        except Exception:
            with open(recommendation_json, 'w') as fh:
                fh.write(repr(recommendation))

        try:
            with open(weather_audit_json, 'w') as fh:
                json.dump(weather_audit, fh, indent=2, default=str)
        except Exception:
            pass

        try:
            with open(growth_stage_audit_json, 'w') as fh:
                json.dump(growth_stage_audit, fh, indent=2, default=str)
        except Exception:
            pass
        # Save collection and STAC search metadata for debugging
        try:
            with open(out_dir / 'stac_collections.json', 'w') as fh:
                json.dump(collections, fh, default=str, indent=2)
        except Exception:
            pass

        try:
            with open(out_dir / 'stac_items.json', 'w') as fh:
                json.dump(stac_items, fh, default=str, indent=2)
        except Exception:
            pass

        # Print concise summary and output locations
        print('\nSaved outputs:')
        print(' -', weather_csv)
        print(' -', satellite_csv)
        print(' -', recommendation_json)

        print('\n=== WEATHER FRAME (head) ===')
        try:
            print(weather_frame.head().to_string())
        except Exception:
            print(repr(weather_frame))

        print('\n=== WEATHER FRAME SHAPE ===')
        try:
            print(getattr(weather_frame, 'shape', None))
        except Exception:
            print('unknown')

            print('\n=== WEATHER REQUEST AUDIT ===')

        try:
            print("Requested Start :", start_date)
            print("Requested End   :", end_date)
            print("Returned Start  :", weather_audit["returned_start"])
            print("Returned End    :", weather_audit["returned_end"])
            print(
                "Range Match     :",
                weather_audit["date_range_matches_request"]
            )
        except Exception as e:
            print(e)

        print('\n=== WEATHER COVERAGE AUDIT ===')

        try:
            for col in weather_audit["coverage_by_column"]:
                coverage = weather_audit["coverage_by_column"][col]
                missing = weather_audit["missing_by_column"][col]

                print(
                    f"{col:<25} "
                    f"coverage={coverage:>6}% "
                    f"missing={missing}"
                )
        except Exception as e:
            print(e)

        print('\n=== SATELLITE SNAPSHOT ===')
        print(satellite_snapshot)

        print('\n=== IRRIGATION RECOMMENDATION ===')
        print(recommendation)
        print('\n=== GROWTH STAGE AUDIT ===')

        try:
            print(
                "Days Since Planting:",
                growth_stage_audit["days_since_planting"]
            )

            print(
                "Cumulative GDD:",
                growth_stage_audit["gdd_cumulative"]
            )

            print(
                "Growth Stage:",
                growth_stage_audit["growth_stage"]
            )

            print(
                "Satellite NDVI:",
                growth_stage_audit["sat_ndvi"]
            )

            if growth_stage_audit["warnings"]:
                print("\nWarnings:")
                for warning in growth_stage_audit["warnings"]:
                    print(" -", warning)

        except Exception as e:
            print(e)

        if not weather_audit["date_range_matches_request"]:
            print(
                "\nWARNING: Weather API did not return the requested "
                "historical date range."
            )

        if growth_stage_audit["warnings"]:
            print(
                "\nWARNING: Growth stage consistency issues detected."
            )
        # Basic sanity checks
        self.assertIsNotNone(weather_frame)
        self.assertIsNotNone(satellite_snapshot)
        self.assertIn('weather_data_summary', recommendation)
        self.assertIn('satellite_data_summary', recommendation)


if __name__ == '__main__':
    unittest.main()
