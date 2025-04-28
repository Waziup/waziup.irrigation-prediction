## Waziup Irrigation Prediction Tests

import json
import requests
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

wazidev_sensor_id = 'temperatureSensor_1'
wazidev_sensor_value = 45.7
wazidev_actuator_id = 'act1'
wazidev_actuator_value = json.dumps(True)

#wazigate_ip = os.environ.get('WAZIGATE_IP', '172.16.11.186')
wazigate_ip = os.environ.get('WAZIGATE_IP', '192.168.188.29')
wazigate_url = 'http://' + wazigate_ip
wazigate_app_url = wazigate_url + "/apps/waziup.irrigation-prediction/api"

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
# GET /api/getValuesForDashboard - Get dashboard values
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
        # Authentication
        resp = requests.post(wazigate_url + '/auth/token', json=auth)
        self.token = resp.text.strip('"')
        self.headers = {'Authorization': f'Bearer {self.token}'}
        self.cookies = {'Token': self.token}

        # Test plot setup
        self.test_plot = {
            'name': 'Test Plot',
            'sensors': ['sensor1', 'sensor2'],
            'config': {
                'threshold': 30,
                'irrigation_amount': 100,
                'sensor_kind': 'tension'
            }
        }

        # Create test plot
        

    # Plot Management Tests
    def test_plot_lifecycle(self):
        # Create plot
        create_resp = requests.post(
            f"{wazigate_app_url}/api/addPlot",
            data={'tab_nr': 25},
            cookies=self.cookies
        )
        self.assertEqual(create_resp.status_code, 200)
        
        # Get plots
        get_resp = requests.get(
            f"{wazigate_app_url}/api/getPlots",
            cookies=self.cookies
        )
        self.assertEqual(get_resp.status_code, 200)
        self.assertGreater(len(get_resp.json()['tabnames']), 0)

        # Remove plot
        remove_resp = requests.post(
            f"{wazigate_app_url}/api/removePlot",
            data={'currentPlot': 25},
            cookies=self.cookies
        )
        self.assertEqual(remove_resp.status_code, 200)

    # Configuration Tests
    def test_config_management(self):
        # Set config
        config_data = {
            'selectedOptionsMoisture': ['sensor1'],
            'selectedOptionsTemp': ['sensor2'],
            'name': 'Test Plot',
            'sensor_kind': 'tension',
            'gps': '52.52,13.405',
            'slope': '5',
            'thres': '30',
            'amount': '100',
            'lookahead': '24',
            'start': '2023-01-01',
            'period': '7'
        }
        set_resp = requests.post(
            f"{wazigate_app_url}/api/setConfig",
            data=config_data,
            cookies=self.cookies
        )
        self.assertEqual(set_resp.status_code, 200)

        # Verify config
        get_resp = requests.get(
            f"{wazigate_app_url}/api/returnConfig",
            cookies=self.cookies
        )
        self.assertEqual(get_resp.status_code, 200)
        config = get_resp.json()
        self.assertEqual(config['data']['Name'], 'Test Plot')

    # Sensor Data Tests
    def test_sensor_data_endpoints(self):
        # Dashboard values
        dash_resp = requests.get(
            f"{wazigate_app_url}/api/getValuesForDashboard",
            cookies=self.cookies
        )
        self.assertEqual(dash_resp.status_code, 200)
        self.assertIn('temp_average', dash_resp.json())

        # Historical data
        hist_resp = requests.get(
            f"{wazigate_app_url}/api/getHistoricalChartData",
            cookies=self.cookies
        )
        self.assertIn(hist_resp.status_code, [200, 404])

    # Prediction Tests
    def test_prediction_workflow(self):
        # Start training
        train_resp = requests.get(
            f"{wazigate_app_url}/api/startTraining",
            cookies=self.cookies
        )
        self.assertEqual(train_resp.status_code, 200)

        # Check training status
        status_resp = requests.get(
            f"{wazigate_app_url}/api/isTrainingReady",
            cookies=self.cookies
        )
        self.assertEqual(status_resp.status_code, 200)

        # Get predictions
        pred_resp = requests.get(
            f"{wazigate_app_url}/api/getPredictionChartData",
            cookies=self.cookies
        )
        self.assertIn(pred_resp.status_code, [200, 404])

    # Irrigation Control Tests
    def test_irrigation_controls(self):
        # Manual irrigation
        irrig_resp = requests.get(
            f"{wazigate_app_url}/api/irrigateManually",
            params={'amount': 50},
            cookies=self.cookies
        )
        self.assertEqual(irrig_resp.status_code, 200)

        # Check irrigation status
        status_resp = requests.get(
            f"{wazigate_app_url}/api/checkActiveIrrigation",
            cookies=self.cookies
        )
        self.assertIn(status_resp.status_code, [200, 404])

    # Edge Cases
    def test_edge_cases(self):
        # Invalid plot ID
        invalid_plot_resp = requests.get(
            f"{wazigate_app_url}/api/getCurrentPlot",
            params={'plotId': 999},
            cookies=self.cookies
        )
        self.assertIn(invalid_plot_resp.status_code, [404, 400])

        # Missing config
        missing_config_resp = requests.get(
            f"{wazigate_app_url}/api/returnConfig",
            cookies=self.cookies
        )
        self.assertIn(missing_config_resp.status_code, [200, 404])

    # Security Tests
    def test_authentication(self):
        # Unauthenticated request
        unauth_resp = requests.get(
            f"{wazigate_app_url}/api/getPlots"
        )
        self.assertIn(unauth_resp.status_code, [401, 403])

    # Performance Tests
    def test_response_times(self):
        endpoints = [
            '/api/getPlots',
            '/api/getValuesForDashboard',
            '/api/getHistoricalChartData'
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            resp = requests.get(
                f"{wazigate_app_url}{endpoint}",
                cookies=self.cookies
            )
            response_time = time.time() - start_time
            self.assertLess(response_time, 2)  # 2 seconds max

    # Conduct Unittests against the Apps API
class TestIrrigationPredictionIntegration(unittest.TestCase):
    token = None
    current_plot_id = None

    def setUp(self):
        # Authentication
        resp = requests.post(wazigate_url + '/auth/token', json=auth)
        self.token = resp.text.strip('"')
        self.headers = {'Authorization': f'Bearer {self.token}'}
        self.cookies = {'Token': self.token}


if __name__ == "__main__":
    unittest.main(
        testRunner=xmlrunner.XMLTestRunner(output='test-reports', verbosity=2),
        failfast=False,
        buffer=False,
        catchbreak=False
    )

