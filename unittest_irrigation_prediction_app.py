## Waziup Irrigation Prediction Tests

import json
from dotenv import dotenv_values, set_key, load_dotenv
import requests
import urllib
import requests_unixsocket
requests_unixsocket.monkeypatch()
from time import sleep
import unittest
import xmlrunner
import random
import logging
import time
import os
import sys
import logging
from xmlrunner import XMLTestRunner

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

## Variable declaration

# Path to the socket (must match your container’s mount)
PROXY_SOCK_PATH = "/var/lib/waziapp/proxy.sock"

# Use the Unix-socket-based base URL
# Note: requests_unixsocket encodes the socket path into the URL
wazigate_base = f"http+unix://{PROXY_SOCK_PATH.replace('/', '%2F')}"
wazigate_app_url = f"{wazigate_base}/api"

wazigate_device = {
  'id': 'test000',
  'name': 'test',
  'sensors': [],
  'actuators': []
}

wazigate_create_actuator = {
  'id': 'act1',
  'name': 'act1'
}

auth = {
  "username": "admin",
  "password": "loragateway"
}

auth_Token_header_Accept_text_plain = {
	'content-type': 'application/json',
	'authorization': 'Bearer **',
    'Accept': 'text/plain'
}

header_Accept_text_plain = {
    'Content-Type' : 'application/json',
    'Accept' : 'text/plain'
}

ENV_FILE = ".env"

# API Endpoints List (overview):
# GET / - Basic endpoint
# GET/POST /ui/(.*) - Serve UI files
# GET /api/getApiUrl - Get API URL
# POST /api/setPlot - Set current plot
# GET /api/getPlots - Get all plots
# POST /api/addPlot - Add new plot
# POST /api/removePlot - Remove plot
# POST /api/setConfig - Set configuration
# GET /api/returnConfig - Get current config
# GET /api/checkConfigPresent - Check config exists
# GET /api/checkActiveIrrigation - Check irrigation status
# GET /api/irrigateManually - Manual irrigation
# GET /api/getValuesForDashboard - Get dashboard values e.g. curl --unix-socket /var/lib/waziapp/proxy.sock http://localhost/api/getValuesForDashboard
# GET /api/getHistoricalChartData - Get historical data
# GET /api/getDatasetChartData - Get dataset data
# GET /api/getPredictionChartData - Get prediction data
# GET /api/getThreshold - Get threshold timestamp
# GET /api/getSensorKind - Get sensor type
# GET /api/isTrainingReady - Check training status
# GET /api/startTraining - Start training
# GET /api/getCurrentPlot - Get current plot ID

# Conduct Unittests against the Apps API
class TestIrrigationPredictionAPI(unittest.TestCase):
    token = None
    current_plot_id = None

    def setUp(self):
        # Load original .env as a backup
        self.original_env = dotenv_values(ENV_FILE)

        # Set temporary values
        set_key(ENV_FILE, "SKIP_TRAINING", "False")
        set_key(ENV_FILE, "LOAD_DATA_FROM_CSV", "True")
        set_key(ENV_FILE, "PERFORM_TRAINING", "True")

        # (Optional) Reload env vars in current process
        load_dotenv(ENV_FILE, override=True)


        # Authentication
        self.session_wg = requests.Session()

        token_url = "http://wazigate/" + "auth/token" # to get out of the container to host, look docker-compose.yml
            
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

        # Send a GET request to the API
        response = requests.get(encoded_url, headers=headers, json=data)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # The response content contains the data from the API
            self.token = response.json().strip('"')
            print("Token retrieved successfully:", self.token)
        else:
            print("Request failed with status code:", response.status_code)

        # Create a unixsocket session with the token
        self.session = requests_unixsocket.Session()
        self.headers = {
            'Cookie': f"Token={self.token}",
            'Content-Type': 'application/x-www-form-urlencoded'
            }
        self.cookies = {'Token': self.token}

        # Load original .env as a backup
        self.original_env = dotenv_values(ENV_FILE)

        # Set temporary values
        set_key(ENV_FILE, "SKIP_TRAINING", "False")
        set_key(ENV_FILE, "LOAD_DATA_FROM_CSV", "True")
        set_key(ENV_FILE, "PERFORM_TRAINING", "True")

        # (Optional) Reload env vars in current process
        load_dotenv(ENV_FILE, override=True)

    def tearDown(self):
        # Restore original .env values
        for key, value in self.original_env.items():
            set_key(ENV_FILE, key, value)

        # Remove keys that were added only for the test
        test_keys = set(dotenv_values(ENV_FILE).keys()) - set(self.original_env.keys())
        for key in test_keys:
            set_key(ENV_FILE, key, "")

    # Evaluate status codes
    def evaluate_status_code(self,statusCode,expected_statusCode):
        if statusCode != expected_statusCode:
            self.setUp()
            print("new token was requested: ", self.token, statusCode)
            return True
        else:
            return False
        

    # Plot Management Tests
    def test_02_test_plot_lifecycle(self):
        # Get plots
        get_resp = self.session.get(
            f"{wazigate_app_url}/getPlots",
            headers=self.headers
        )
        self.assertEqual(get_resp.status_code, 200)
        self.assertGreater(len(get_resp.json()['tabnames']), 0)
        print(get_resp.text)
        next_plot_to_add = len(get_resp.json()['tabnames']) + 1
        print("Next plot to add: ", next_plot_to_add)

        # Create plot
        print("Creating plot...")
        create_resp = self.session.post(
            f"{wazigate_app_url}/addPlot",
            data={'tab_nr': next_plot_to_add},
            headers=self.headers
        )
        self.assertEqual(create_resp.status_code, 200)
        print(create_resp.text)
        plot_recently_created = int(create_resp.json()["plot_number"])

        # Set current plot
        print("Set plot...")
        set_resp = self.session.post(
            f"{wazigate_app_url}/setPlot",
            data=f"currentPlot={next_plot_to_add}",
            headers=self.headers
        )
        self.assertEqual(set_resp.status_code, 200)
        print(set_resp.text)

        # Remove plot -> HTTP code fails, but is deleted
        remove_resp = self.session.post(
            f"{wazigate_app_url}/removePlot",
            data=f"currentPlot={next_plot_to_add}",
            headers=self.headers
        )
        self.assertEqual(remove_resp.status_code, 200)

    # Configuration Tests
    def test_03_test_config_management(self):
        # Get plots
        get_resp = self.session.get(
            f"{wazigate_app_url}/getPlots",
            headers=self.headers
        )
        self.assertEqual(get_resp.status_code, 200)
        self.assertGreater(len(get_resp.json()['tabnames']), 0)
        print(get_resp.text)
        next_plot_to_add = len(get_resp.json()['tabnames']) + 1
        print("Next plot to add: ", next_plot_to_add)

        # Create plot
        print("Creating plot...")
        create_resp = self.session.post(
            f"{wazigate_app_url}/addPlot",
            data={'tab_nr': next_plot_to_add},
            headers=self.headers
        )
        self.assertEqual(create_resp.status_code, 200)
        #print(create_resp.text)
        plot_recently_created = int(create_resp.json()["plot_number"])

        # Set config
        config_data = {
            'selectedOptionsMoisture': ['657360e268f319085542e336/657360f468f319085542e337'],
            'selectedOptionsTemp': ['657360e268f319085542e336/657360f468f319085542e33b'],
            'selectedOptionsFlow': ['641b13011d41c8b9a2682344/641b13101d41c8b9a2682345'],
            'name': 'Test Plot',
            'sensor_kind': 'tension',
            'gps': ['51.023591, 13.744087'],
            'slope': '5',
            'thres': '30',
            'amount': '100',
            'lookahead': '24',
            'start': '2025-03-11T00:00:00.000Z',
            'period': '7',
            'soil': 'Sandy_Soil',
            'pwp': '3',
            'fcu': '2',
            'fcl': '1',
            'sat': '0',
            'ret': ['Soil tension,VWC\n0, 0.225\n5, 0.2\n10, 0.185\n20, 0.15\n50, 0.125\n100, 0.1\n200, 0.075\n500, 0.05\n1000, 0.025\n']
        }

        set_resp = self.session.post(
            f"{wazigate_app_url}/setConfig",
            data=config_data,
            headers=self.headers
        )
        self.assertEqual(set_resp.status_code, 200)

        # Get config back
        get_resp = self.session.get(
            f"{wazigate_app_url}/returnConfig",
            headers=self.headers
        )
        self.assertEqual(get_resp.status_code, 200)

        # Parse returned config
        config = get_resp.json()['data']
        #print(json.dumps(config, indent=2))  # for debugging

        # Match returned field names and types
        self.assertEqual(config['Name'], 'Test Plot')
        self.assertEqual(config['Sensor_kind'], 'tension')
        self.assertEqual(config['Gps_info']['lattitude'], '51.023591')
        self.assertEqual(config['Gps_info']['longitude'], '13.744087')
        self.assertAlmostEqual(float(config['Slope']), 5.0)
        self.assertAlmostEqual(float(config['Threshold']), 30.0)
        self.assertAlmostEqual(float(config['Irrigation_amount']), 100.0)
        self.assertAlmostEqual(float(config['Look_ahead_time']), 24.0)
        self.assertEqual(config['Start_date'], '2025-03-11T00:00:00.000Z')
        self.assertEqual(int(config['Period']), 7)
        self.assertEqual(config['Soil_type'], 'Sandy_Soil')
        self.assertEqual(float(config['PermanentWiltingPoint']), 3.0)
        self.assertEqual(float(config['FieldCapacityUpper']), 2.0)
        self.assertEqual(float(config['FieldCapacityLower']), 1.0)
        self.assertEqual(float(config['Saturation']), 0.0)

        # Sensor ID checks
        self.assertIn('657360e268f319085542e336/657360f468f319085542e337', config['DeviceAndSensorIdsMoisture'][0])
        self.assertIn('657360e268f319085542e336/657360f468f319085542e33b', config['DeviceAndSensorIdsTemp'][0])
        self.assertIn('641b13011d41c8b9a2682344/641b13101d41c8b9a2682345', config['DeviceAndSensorIdsFlow'][0])

        # Check CSV was parsed and converted to list of dicts
        retention = config.get('Soil_water_retention_curve', [])
        self.assertTrue(any("Soil tension" in entry and "VWC" in entry for entry in retention))

    # Sensor Data Tests
    def test_04_test_sensor_data_endpoints(self):
        # Dashboard values
        dash_resp = self.session.get(
            f"{wazigate_app_url}/getValuesForDashboard",
            headers=self.headers
        )
        self.assertEqual(dash_resp.status_code, 200)
        self.assertIn('temp_average', dash_resp.json())

        # Historical data
        hist_resp = self.session.get(
            f"{wazigate_app_url}/getHistoricalChartData",
            headers=self.headers
        )
        self.assertIn(hist_resp.status_code, [200, 404])

    # Prediction Tests    
    def test_05_test_prediction_workflow(self):
        # Start training
        train_resp = self.session.get(
            f"{wazigate_app_url}/startTraining",
            headers=self.headers
        )
        self.assertEqual(train_resp.status_code, 200)
        print("Training started...")

         # Poll until training is finished
        max_wait_time = 3600  # seconds (one hour)
        interval = 10  # poll every 10 seconds
        waited = 0

        while waited < max_wait_time:
            time.sleep(interval)
            waited += interval

            status_resp = self.session.get(
                f"{wazigate_app_url}/isTrainingReady",
                headers=self.headers
            )
            self.assertEqual(status_resp.status_code, 200)

            try:
                ready = status_resp.json().get("isTrainingFinished", False)
                print(f"Polling status after {waited}s: isTrainingFinished = {ready}")
            except Exception as e:
                self.fail(f"Invalid JSON from isTrainingReady: {e}")

            if ready:
                print(f"Training finished after {waited} seconds.")
                break
        else:
            self.fail(f"Training did not finish within {max_wait_time} seconds.")

        # Afterwards get predictions
        pred_resp = self.session.get(
            f"{wazigate_app_url}/getPredictionChartData",
            headers=self.headers
        )
        self.assertIn(pred_resp.status_code, 200)

    # Irrigation Control Tests
    def test_06_test_irrigation_controls(self):
        # Manual irrigation
        irrig_resp = self.session.get(
            f"{wazigate_app_url}/irrigateManually",
            params={'amount': 50},
            headers=self.headers
        )
        self.assertEqual(irrig_resp.status_code, 200)

        # Check irrigation status
        status_resp = self.session.get(
            f"{wazigate_app_url}/checkActiveIrrigation",
            headers=self.headers
        )
        self.assertIn(status_resp.status_code, [200, 404])

    # Edge Cases
    def test_edge_cases(self):
        # Invalid plot ID
        invalid_plot_resp = self.session.get(
            f"{wazigate_app_url}/getCurrentPlot",
            params={'plotId': 999},
            headers=self.headers
        )
        self.assertIn(invalid_plot_resp.status_code, [404, 400])

        # Missing config
        missing_config_resp = self.session.get(
            f"{wazigate_app_url}/returnConfig",
            headers=self.headers
        )
        self.assertIn(missing_config_resp.status_code, [200, 404])

    # Security Tests -> useless because API is not protected (locally)
    def test_09_test_authentication(self):
        # Unauthenticated request
        unauth_resp = self.session.get(
            f"{wazigate_app_url}/getPlots"
        )
        self.assertIn(unauth_resp.status_code, [401, 403])

    # Performance Tests
    def test_10_test_response_times(self):
        endpoints = [
            '/api/getPlots',
            '/api/getValuesForDashboard',
            '/api/getHistoricalChartData'
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            resp = self.session.get(
                f"{wazigate_app_url}{endpoint}",
                cookies=self.headers
            )
            response_time = time.time() - start_time
            self.assertLess(response_time, 60)  # 10 seconds max

#     # Conduct Unittests against the Apps API
# class TestIrrigationPredictionIntegration(unittest.TestCase):
#     token = None
#     current_plot_id = None

#     def setUp(self):
#         # Authentication
#         resp = self.session.post(wazigate_url + '/auth/token', json=auth)
#         self.token = resp.text.strip('"')
#         self.headers = {'Authorization': f'Bearer {self.token}'}
#         self.cookies = {'Token': self.token}


if __name__ == "__main__":
    # Ensure the directories exists
    os.makedirs("/root/src/tests", exist_ok=True)
    os.makedirs("/root/src/test-reports", exist_ok=True)
    # Run the tests with XML output
    unittest.main(
        testRunner=xmlrunner.XMLTestRunner(output='test-reports', verbosity=2),
        failfast=False,
        buffer=False,
        catchbreak=False
    )

