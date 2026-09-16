import datetime
import os
from dotenv import load_dotenv
import pytz
#from geopy.geocoders import Nominatim
import requests
import urllib.parse
import re

class TimeUtils:
    # Deprecated compatibility attribute. Production code resolves timezone
    # from each plot and must never use mutable process-wide farm state.
    Timezone = ''

    @staticmethod
    def for_plot(plot, fallback="UTC"):
        """Return a validated per-plot timezone without sharing farm state."""
        value = getattr(plot, "timezone", None) or fallback
        try:
            pytz.timezone(value)
        except pytz.UnknownTimeZoneError as exc:
            raise ValueError(f"Invalid timezone {value!r}") from exc
        return value

    @staticmethod
    def get_timezone_offset(timezone_str):
        """
        Returns the UTC offset (in hours) for the given timezone string.
        """
        timezone = pytz.timezone(timezone_str)
        current_time = datetime.datetime.now(tz=timezone)
        utc_offset = current_time.utcoffset().total_seconds() / 3600.0
        return utc_offset

    @staticmethod
    def get_timezone(latitude_str, longitude_str):
        """
        Returns the timezone string for the given latitude and longitude.
        """
        # Convert to floats
        latitude = float(latitude_str)
        longitude = float(longitude_str)

        # Get location data using geopy
        #geolocator = Nominatim(user_agent="timezone_finder", timeout=5)
        #location = geolocator.reverse((latitude, longitude), language="en")
        
        # Determine the timezone using TimezoneFinder
        from timezonefinder import TimezoneFinder
        timezone_finder = TimezoneFinder()
        timezone_str = timezone_finder.timezone_at(lng=longitude, lat=latitude)
        
        return timezone_str
    
class NetworkUtils:
    # Stores all env vars
    Env = os.environ

    # Specific network related properties 
    ApiUrl = ""
    Proxy = ""
    Token = ""
    
    # Rertieve from .env file
    @classmethod
    def get_env(cls):
        load_dotenv()
        api_url = (cls.Env.get("API_URL") or "").strip()
        if not api_url:
            raise RuntimeError("API_URL is required")
        parsed = urllib.parse.urlsplit(api_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError("API_URL must be an absolute HTTP(S) URL")
        cls.ApiUrl = api_url.rstrip("/") + "/"
        cls.Proxy = (cls.Env.get("Proxy_URL") or "").strip()

    @classmethod
    def is_local_gateway(cls):
        """Return whether the configured API is the local WaziGate service."""
        hostname = (urllib.parse.urlsplit(cls.ApiUrl).hostname or "").lower()
        return hostname in {"wazigate", "localhost", "127.0.0.1", "::1"}

    @classmethod
    def get_gateway_id(cls):
        """Read the configured gateway's identity from WaziGate /device/id."""
        if not cls.ApiUrl:
            raise ValueError("Gateway API is not configured")
        headers = {"Authorization": f"Bearer {cls.Token}"} if cls.Token else {}
        response = requests.get(cls.ApiUrl.rstrip("/") + "/device/id",
                                headers=headers, timeout=10)
        response.raise_for_status()
        gateway_id = response.text.strip().strip('"')
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", gateway_id):
            raise ValueError("Gateway returned an invalid ID")
        return gateway_id


    @classmethod
    def get_token(cls):
        """Obtain a remote-gateway token without embedding or logging secrets."""
        if not cls.ApiUrl:
            raise RuntimeError("NetworkUtils.get_env() must run before get_token()")
        if cls.is_local_gateway():
            cls.Token = ""
            print('There is no token needed, fetch data from local gateway.')
            return ""

        username = (cls.Env.get("WAZIGATE_USERNAME") or "").strip()
        password = cls.Env.get("WAZIGATE_PASSWORD") or ""
        if not username or not password:
            raise RuntimeError(
                "WAZIGATE_USERNAME and WAZIGATE_PASSWORD are required for a remote gateway"
            )

        token_url = urllib.parse.urljoin(cls.ApiUrl, "auth/token")
        response = requests.post(
            token_url,
            headers={"accept": "application/json"},
            json={"username": username, "password": password},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, str):
            token = payload
        elif isinstance(payload, dict):
            token = payload.get("token") or payload.get("access_token")
        else:
            token = None
        if not isinstance(token, str) or not token.strip():
            raise RuntimeError("Gateway token response did not contain a token")
        cls.Token = token.strip()
        print("Gateway token retrieved successfully.")
        return cls.Token
