import datetime
import os
from dotenv import load_dotenv
import pytz
from geopy.geocoders import Nominatim
import requests
from timezonefinder import TimezoneFinder
import urllib.parse

class TimeUtils:
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
        geolocator = Nominatim(user_agent="timezone_finder")
        location = geolocator.reverse((latitude, longitude), language="en")
        
        # Determine the timezone using TimezoneFinder
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
        cls.ApiUrl = cls.Env.get("API_URL")
        cls.Proxy = cls.Env.get("Proxy_URL")


    # Not ready: error handling, Token needs to be renewed (not that important as only for remote gw =>debug)
    @classmethod
    def get_token(cls):
        # Generate token to fetch data from another gateway
        if cls.ApiUrl.startswith('http://wazigate/'):
            print('There is no token needed, fetch data from local gateway.')
        # Get token, important for non localhost devices
        else:
            # curl -X POST "http://192.168.189.2/auth/token" -H "accept: application/json" -d "{\"username\": \"admin\", \"password\": \"loragateway\"}"
            token_url = cls.ApiUrl + "auth/token"
            
            # Parse the URL
            parsed_token_url = urllib.parse.urlsplit(token_url)
            
            # Encode the query parameters
            encoded_query = urllib.parse.quote(parsed_token_url.query, safe='=&')
            
            # Reconstruct the URL with the encoded query
            encoded_url = urllib.parse.urlunsplit((parsed_token_url.scheme, 
                                                parsed_token_url.netloc, 
                                                parsed_token_url.path, 
                                                encoded_query, 
                                                parsed_token_url.fragment))
            
            # Define headers for the POST request
            headers = {
                'accept': 'application/json',
                #'Content-Type': 'application/json',  # Make sure to set Content-Type
            }
            
            # Define data for the GET request
            data = {
                'username': 'admin',
                'password': 'loragateway',
            }

            try:
                # Send a GET request to the API
                response = requests.post(encoded_url, headers=headers, json=data)

                # Check if the request was successful (status code 200)
                if response.status_code == 200:
                    # The response content contains the data from the API
                    cls.Token = response.json()
                else:
                    print("Request failed with status code:", response.status_code)
            except requests.exceptions.RequestException as e:
                # Handle request exceptions (e.g., connection errors)
                print("Request error:", e)
                
                return "", e #TODO: intruduce error handling!