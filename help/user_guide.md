TODO: NOT READY... just made up with chat-gpt, did some edits
---------------
# Irrigation Prediction Application User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Setup and Configuration](#setup-and-configuration)
5. [Using the Application](#using-the-application)
   - [Connecting Sensors](#connecting-sensors)
   - [Training the Prediction Model](#training-the-prediction-model)
   - [Adjusting Settings](#adjusting-settings)
6. [Understanding Predictions](#understanding-predictions)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

---

## Introduction
The **Irrigation Prediction Application** is designed to predict soil moisture levels, supporting efficient water management by integrating with **WaziGate** and connected soil moisture sensors.

## System Requirements
- **Supported OS:** WaziGateOS
- **Required Hardware:** WaziGate, soil moisture sensor(s), temperature sensor(s)
- **Optional Hardware:** Flow meter and a pump actuated with relay, to conduct the actuation
- **Software Dependencies:** None, comes as docker container with all dependencies included
- **Internet Connection:** Required for retriving weather data and model updates


Installation
-------
Visit the WaziGates UI and open the App section. Press the plus button and click install custom app. Type into the textbox "waziup/irrigation-prediction:dev" to download/install the application from dockerhub. 

All dependencies are included in the docker image, so no further actions are required.
Configuring Sensor Settings
Describe how to connect soil moisture sensors to the WaziGate.

Connecting to Local Wi-Fi
------------------------
Explain how to connect the WaziGate to local Wi-Fi or how to configure the device’s hotspot settings if necessary.

Using the Application

Explaination of Different Attributes:
-------------------------------------
- ***Soil Moisture Sensor Selection***: Select one or more soil moisture sensors that monitor soil tension and are connected to your WaziGate. To select or deselect multiple sensors, hold down the **CTRL** key.

- ***Soil Temperature Sensor Selection***: Choose one or more sensors that measure soil temperature and are linked to your WaziGate. Use **CTRL** to select or deselect multiple sensors.

- ***Water Flow Sensor***: Select the sensor that monitors the water flow of your pump connected to WaziGate. Only one water flow sensor can be chosen.

- ***GPS Coordinates***: Enter coordinates to fetch relevant meteorological data from online sources.

- ***Slope Detection***: Specify the slope of your field to assist in detecting artificial irrigation. This option is needed only if no water flow sensor is added.

- ***Irrigation Volume***: Enter the volume of water (in liters) used for a single irrigation event.

- ***Forecast Look-Ahead Time***: Set the number of hours ahead for which you want soil tension forecasts.

- ***Data Start Date***: Select the start date for sensor data to be included in model creation. It is recommended to allow a short warm-up period after sensor installation.

- ***Maximum Data Duration***: Enter the maximum period (in days) of data to include in the model.

- ***Soil Type***: Choose the soil type that best matches your field’s composition.

- ***Soil Water Retention Data***: Input soil water retention curve data in key-value pairs to help in prediction accuracy.

- ***Permanent Wilting Point (PWP)***: Enter the soil moisture content at which plants can no longer extract water from the soil.

- ***Field Capacity Upper Limit (FCU)***: Input the maximum soil moisture content that your soil can retain.

- ***Field Capacity Lower Limit (FCL)***: Input the minimum soil moisture content that your soil can hold.

- ***Soil Saturation (SAT)***: Enter the moisture content level for when the soil is fully saturated.

- ***Soil Tension Threshold***: Specify the threshold for soil tension, measured in cbar or hPa, to guide irrigation decisions.

### Overview Tooltips
- ***Sensor Overview***: Displays real-time data from connected sensors.

- ***Manual Irrigation Control***: Allows manual triggering of irrigation.

- ***Historical Data Chart***: Displays raw sensor data, with a 1,000-element limit for faster performance.

- ***Training Data Chart***: Shows the full data set used to train the machine learning model.

- ***Prediction Chart***: Provides the latest soil tension forecast for the upcoming week.



Verify Sensor Data
Explain how to check sensor status and readouts.

Training the Prediction Model
Open the Training Tab
Explain how users can start the training process and what they should expect.

Adjust Model Parameters

Available hyperparameters and how they impact predictions.
Approximate training time.
Adjusting Settings
List and explain the available settings, such as model type, prediction intervals, and data refresh rates.

Understanding Predictions
Describe how to interpret predictions in the application. Are predictions presented as simple values or through a visual format, like graphs? How does the app indicate whether soil moisture is adequate?

Troubleshooting
Error Messages: Explain common errors users might encounter and potential fixes.
Connectivity Issues: Steps to take if sensors aren’t connecting or if there’s a data lag.
FAQ
How frequently should I re-train the model? This option is hidden from the user, it needs to integrated into config...
Can I use different types of sensors?
How do I access historical predictions?