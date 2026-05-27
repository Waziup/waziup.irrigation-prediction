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

        # Fetch weather for the period 2026-01-05 .. 2026-05-18 (user requested)
        start_date = planting_date
        end_date = '2026-05-18'

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
            irrigation_type='drip',
            plot_area_m2=1000.0,
            initial_gdd=0.0,
        )

        recommendation = get_irrigation_recommendation(plot)

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

        print('\n=== SATELLITE SNAPSHOT ===')
        print(satellite_snapshot)

        print('\n=== IRRIGATION RECOMMENDATION ===')
        print(recommendation)

        # Basic sanity checks
        self.assertIsNotNone(weather_frame)
        self.assertIsNotNone(satellite_snapshot)
        self.assertIn('weather_data_summary', recommendation)
        self.assertIn('satellite_data_summary', recommendation)


if __name__ == '__main__':
    unittest.main()
