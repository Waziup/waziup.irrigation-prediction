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
#wazigate_ip = os.environ.get('WAZIGATE_IP', '192.168.188.29')
wazigate_ip = os.environ.get('WAZIGATE_IP', 'localhost:8080')
wazigate_url = 'http://' + wazigate_ip
wazigate_app_url = wazigate_url + "/apps/waziup.irrigation-prediction"

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
        self.headers = {
            'Cookie': f"Token={self.token}",
            'Content-Type': 'application/x-www-form-urlencoded'
            }
        self.cookies = {'Token': self.token}

        # # Test plot setup
        # self.test_plot = {
        #     'name': 'Test Plot',
        #     'sensors': ['sensor1', 'sensor2'],
        #     'config': {
        #         'threshold': 30,
        #         'irrigation_amount': 100,
        #         'sensor_kind': 'tension'
        #     }
        # }

        # Create test plot

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
        # Create plot
        print("Creating plot...")
        create_resp = requests.post(
            f"{wazigate_app_url}/api/addPlot",
            data={'tab_nr': 25},
            headers=self.headers
        )
        self.assertEqual(create_resp.status_code, 200)
        print(create_resp.text)
        plot_recently_created = int(create_resp.json()["plot_number"])
        
        # Get plots
        get_resp = requests.get(
            f"{wazigate_app_url}/api/getPlots",
            headers=self.headers
        )
        self.assertEqual(get_resp.status_code, 200)
        self.assertGreater(len(get_resp.json()['tabnames']), 0)
        print(get_resp.text)

        # Set current plot
        print("Set plot...")
        set_resp = requests.post(
            f"{wazigate_app_url}/api/setPlot",
            data=f"currentPlot={plot_recently_created}",
            headers=self.headers
        )
        self.assertEqual(set_resp.status_code, 200)
        print(set_resp.text)

        # Remove plot -> HTTP code fails, but is deleted
        remove_resp = requests.post(
            f"{wazigate_app_url}/api/removePlot",
            data=f"currentPlot={plot_recently_created}",
            headers=self.headers
        )
        self.assertEqual(remove_resp.status_code, 200)

    # Configuration Tests
    def test_03_test_config_management(self):
        # Create plot
        print("Creating plot...")
        create_resp = requests.post(
            f"{wazigate_app_url}/api/addPlot",
            data={'tab_nr': 25},
            headers=self.headers
        )
        self.assertEqual(create_resp.status_code, 200)
        #print(create_resp.text)
        plot_recently_created = int(create_resp.json()["plot_number"])

        # Set config
        config_data = {
            'selectedOptionsMoisture': ['657360e268f319085542e336/657360f468f319085542e337'],
            'selectedOptionsTemp': ['657360e268f319085542e336/657360f468f319085542e33b'],
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

        set_resp = requests.post(
            f"{wazigate_app_url}/api/setConfig",
            data=config_data,
            headers=self.headers
        )
        self.assertEqual(set_resp.status_code, 200)

        # Get config back
        get_resp = requests.get(
            f"{wazigate_app_url}/api/returnConfig",
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

        # Check CSV was parsed and converted to list of dicts
        retention = config.get('Soil_water_retention_curve', [])
        self.assertTrue(any("Soil tension" in entry and "VWC" in entry for entry in retention))

    # Sensor Data Tests
    def test_04_test_sensor_data_endpoints(self):
        # Dashboard values
        dash_resp = requests.get(
            f"{wazigate_app_url}/api/getValuesForDashboard",
            headers=self.headers
        )
        self.assertEqual(dash_resp.status_code, 200)
        self.assertIn('temp_average', dash_resp.json())

        # Historical data
        hist_resp = requests.get(
            f"{wazigate_app_url}/api/getHistoricalChartData",
            headers=self.headers
        )
        self.assertIn(hist_resp.status_code, [200, 404])

    # Prediction Tests
    def test_05_test_prediction_workflow(self):
        # Start training
        train_resp = requests.get(
            f"{wazigate_app_url}/api/startTraining",
            headers=self.headers
        )
        self.assertEqual(train_resp.status_code, 200)

    #     # Check training status
    #     status_resp = requests.get(
    #         f"{wazigate_app_url}/api/isTrainingReady",
    #         headers=self.headers
    #     )
    #     self.assertEqual(status_resp.status_code, 200)

    #     # Get predictions
    #     pred_resp = requests.get(
    #         f"{wazigate_app_url}/api/getPredictionChartData",
    #         headers=self.headers
    #     )
    #     self.assertIn(pred_resp.status_code, [200, 404])

    # # Irrigation Control Tests
    # def test_irrigation_controls(self):
    #     # Manual irrigation
    #     irrig_resp = requests.get(
    #         f"{wazigate_app_url}/api/irrigateManually",
    #         params={'amount': 50},
    #         headers=self.headers
    #     )
    #     self.assertEqual(irrig_resp.status_code, 200)

    #     # Check irrigation status
    #     status_resp = requests.get(
    #         f"{wazigate_app_url}/api/checkActiveIrrigation",
    #         headers=self.headers
    #     )
    #     self.assertIn(status_resp.status_code, [200, 404])

    # # Edge Cases
    # def test_edge_cases(self):
    #     # Invalid plot ID
    #     invalid_plot_resp = requests.get(
    #         f"{wazigate_app_url}/api/getCurrentPlot",
    #         params={'plotId': 999},
    #         headers=self.headers
    #     )
    #     self.assertIn(invalid_plot_resp.status_code, [404, 400])

    #     # Missing config
    #     missing_config_resp = requests.get(
    #         f"{wazigate_app_url}/api/returnConfig",
    #         headers=self.headers
    #     )
    #     self.assertIn(missing_config_resp.status_code, [200, 404])

    # Security Tests -> useless because API is not protected
    def test_06_test_authentication(self):
        # Unauthenticated request
        unauth_resp = requests.get(
            f"{wazigate_app_url}/api/getPlots"
        )
        self.assertIn(unauth_resp.status_code, [401, 403])

    # Performance Tests
    def test_07_test_response_times(self):
        endpoints = [
            '/api/getPlots',
            '/api/getValuesForDashboard',
            '/api/getHistoricalChartData'
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            resp = requests.get(
                f"{wazigate_app_url}{endpoint}",
                cookies=self.headers
            )
            response_time = time.time() - start_time
            self.assertLess(response_time, 5)  # 5 seconds max

#     # Conduct Unittests against the Apps API
# class TestIrrigationPredictionIntegration(unittest.TestCase):
#     token = None
#     current_plot_id = None

#     def setUp(self):
#         # Authentication
#         resp = requests.post(wazigate_url + '/auth/token', json=auth)
#         self.token = resp.text.strip('"')
#         self.headers = {'Authorization': f'Bearer {self.token}'}
#         self.cookies = {'Token': self.token}


if __name__ == "__main__":
    unittest.main(
        testRunner=xmlrunner.XMLTestRunner(output='test-reports', verbosity=2),
        failfast=False,
        buffer=False,
        catchbreak=False
    )

