import os
import requests
import json
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
api_key = os.getenv("SPACEIOTBOX_API_KEY")
base_url = os.getenv("SPACEIOTBOX_BASE_URL", "https://www.smartafrihub.com/spaceiotbox/api").rstrip("/")
headers = {"Accept": "application/json", "X-API-Key": api_key}

coords = [
    ("Mwanza", -2.52, 32.90),
    ("Kisumu", -0.09, 34.75),
    ("Jinja", 0.44, 33.20)
]

endpoints = [
    ("/v1/agro_climate/land", {}),
    ("/v1/agro_climate/water", {}),
]

def analyze_json(data):
    ts_list = []
    def recurse(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ['timestamp', 'date', 'time', 'datetime'] and isinstance(v, str):
                    try:
                        dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
                        ts_list.append(dt)
                    except: pass
                recurse(v)
        elif isinstance(obj, list):
            for item in obj: recurse(item)
    recurse(data)
    if not ts_list: return "No TS found"
    filtered = [t for t in ts_list if datetime(2026, 5, 1) <= t.replace(tzinfo=None) <= datetime(2026, 5, 18)]
    return f"TS: {min(ts_list).isoformat()} to {max(ts_list).isoformat()} ({len(ts_list)} total, {len(filtered)} in range)"

results = []
def call_api(path, params=None):
    try:
        resp = requests.get(f"{base_url}{path}", params=params, headers=headers, timeout=10)
        status = resp.status_code
        text = resp.text[:140].replace("\n", " ")
        analysis = ""
        if status == 200:
            try: analysis = analyze_json(resp.json())
            except: analysis = "JSON error"
        return f"Status {status} | {analysis} | {text}"
    except Exception as e:
        return f"Error: {str(e)}"

for name, lat, lon in coords:
    for path, p in endpoints:
        query = p.copy()
        query.update({"lat": lat, "lon": lon})
        res = call_api(path, query)
        results.append(f"{name} {path} {p}: {res}")

for r in results: print(r)
