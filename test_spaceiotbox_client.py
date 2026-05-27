import json
import os
import unittest
from datetime import date

import pandas as pd
import requests
from dotenv import load_dotenv


BASE_URL = os.getenv("SPACEIOTBOX_BASE_URL", "https://www.smartafrihub.com/spaceiotbox/api").rstrip("/")

# Lake Victoria Basin coordinates within the documented bounds.
LAKE_VICTORIA_POINTS = [
    {"name": "Mwanza", "lat": -2.52, "lon": 32.90},
    {"name": "Kisumu", "lat": -0.09, "lon": 34.75},
    {"name": "Jinja", "lat": 0.44, "lon": 33.20},
]

WINDOW_START = date(2026, 5, 1)
WINDOW_END = date(2026, 5, 18)


def _collect_timestamps(payload, timestamps, field_names):
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_lower = str(key).lower()
            if any(token in key_lower for token in ("time", "date", "timestamp")):
                if isinstance(value, (str, int, float, pd.Timestamp)):
                    parsed = pd.to_datetime(value, errors="coerce", utc=True)
                    if not pd.isna(parsed):
                        timestamps.append(parsed)
                        field_names.add(str(key))
            _collect_timestamps(value, timestamps, field_names)
        return

    if isinstance(payload, list):
        for value in payload:
            _collect_timestamps(value, timestamps, field_names)


class TestSpaceIoTBoxLiveAvailability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(override=False)
        cls.api_key = os.getenv("SPACEIOTBOX_API_KEY", "").strip()
        if not cls.api_key:
            raise unittest.SkipTest("SPACEIOTBOX_API_KEY is not set")

    def _get_json(self, path, params=None):
        response = requests.get(
            f"{BASE_URL}{path}",
            params=params,
            headers={"Accept": "application/json", "X-API-Key": self.api_key},
            timeout=40,
        )
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = {"_raw": response.text[:500]}
        return response, body

    def _assert_endpoint_responds(self, path, params=None):
        response, body = self._get_json(path, params=params)
        self.assertEqual(
            response.status_code,
            200,
            msg=f"{path} returned {response.status_code} body={str(body)[:350]}",
        )
        return body

    def _print_window_stats(self, endpoint_name, payload):
        timestamps = []
        fields = set()
        _collect_timestamps(payload, timestamps, fields)
        if not timestamps:
            print(f"{endpoint_name}: no timestamps found")
            return

        min_ts = min(timestamps)
        max_ts = max(timestamps)
        in_window = sum(
            1
            for ts in timestamps
            if WINDOW_START <= ts.date() <= WINDOW_END
        )
        print(
            f"{endpoint_name}: fields={sorted(fields)} min={min_ts.isoformat()} "
            f"max={max_ts.isoformat()} window_count={in_window}"
        )

    def test_01_agro_climate_land_response_for_window(self):
        for point in LAKE_VICTORIA_POINTS:
            path = "/v1/agro_climate/land"
            base_params = {"lat": point["lat"], "lon": point["lon"]}

            body_no_dates = self._assert_endpoint_responds(path, params=base_params)
            self.assertIsInstance(body_no_dates, dict)
            self.assertIn("forecast_data", body_no_dates)
            self.assertIn("data", body_no_dates)
            vegetation = body_no_dates.get("data", {}).get("vegetation_indices", {})
            self.assertIn("NDVI", vegetation)
            self._print_window_stats(f"{path} {point['name']} no_dates", body_no_dates)

            date_params = {
                **base_params,
                "start_date": WINDOW_START.isoformat(),
                "end_date": WINDOW_END.isoformat(),
            }
            response, body_dates = self._get_json(path, params=date_params)
            self.assertIn(
                response.status_code,
                (200, 422),
                msg=(
                    f"{path} with date params returned {response.status_code} "
                    f"body={str(body_dates)[:350]}"
                ),
            )
            self._print_window_stats(f"{path} {point['name']} with_dates", body_dates)

        water_response, water_body = self._get_json("/v1/agro_climate/water", params={"lat": LAKE_VICTORIA_POINTS[0]["lat"], "lon": LAKE_VICTORIA_POINTS[0]["lon"]})
        self.assertIn(water_response.status_code, (200, 400))
        print(f"/v1/agro_climate/water status={water_response.status_code} body={str(water_body)[:200]}")

    def test_02_eo_locations_and_stac_collections_respond(self):
        locations = self._assert_endpoint_responds("/v1/eo/locations")
        print(f"/v1/eo/locations keys={list(locations.keys()) if isinstance(locations, dict) else 'list'}")

        collections = self._assert_endpoint_responds("/v1/eo/stac/collections")
        if isinstance(collections, dict):
            values = collections.get("collections", [])
        elif isinstance(collections, list):
            values = collections
        else:
            values = []

        ids = []
        for item in values[:5]:
            if isinstance(item, dict) and item.get("id"):
                ids.append(item["id"])
        print(f"/v1/eo/stac/collections count={len(values)} first_ids={ids}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
