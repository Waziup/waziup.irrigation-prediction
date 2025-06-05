import threading
import time
import pathlib
import pickle
from datetime import datetime, timedelta

# local
import create_model
import actuation

Restart_time = 1800 # DEBUG 1800 ~ 30 min in s


class PredictionThread(threading.Thread):
    def __init__(self, plot, name=None):
        super().__init__(name=name)
        self.daemon = True
        self.currentPlot = plot  # Attach thread to a specific plot
        self.stop_event = threading.Event()  # Stop flag

    @staticmethod
    def time_until_n_hours(hours):   # TODO: Put in TimeUtils ASAP
        """Calculate the time difference from now until the next noon."""
        now = datetime.now()#DEBUG
        predict_time = now + timedelta(hours=0, minutes=hours, seconds=0, microseconds=0) #TODO: change to hours DEBUG

        return (predict_time - now).total_seconds()

    def run(self):
        # To stop via event
        while not self.stop_event.is_set():

            # Initial waiting, after model was trained, prediction was conducted and actuation was triggered 
            time_to_sleep = self.time_until_n_hours(self.currentPlot.predict_period_hours)
            print(f"Waiting {time_to_sleep // 3600:.0f} hours {time_to_sleep % 3600 // 60:.0f} minutes until conducting next prediction...")
            time.sleep(time_to_sleep)  # Sleep until threshold
    
            try:
                start_time = datetime.now().replace(microsecond=0)
                print("Prediction started at:", start_time)

                file_path = pathlib.Path('saved_variables_plot_' + str(self.currentPlot.id) +'.pkl')

                if create_model.Perform_training: #same var is used here to preserve functionality
                    # Call predict_with_updated_data function
                    currentSoilTension, self.currentPlot.threshold_timestamp, self.currentPlot.predictions = create_model.predict_with_updated_data(self.currentPlot)

                    # Create object to save
                    variables_to_save = {
                        'currentSoilTension': currentSoilTension,
                        'threshold_timestamp': self.currentPlot.threshold_timestamp,
                        'predictions': self.currentPlot.predictions
                    }
                    # Save the variables to a file
                    with open(file_path, 'wb') as f:
                        pickle.dump(variables_to_save, f)
                else:
                    # Load the saved variables from the file
                    with open(file_path, 'rb') as f:
                        loaded_variables = pickle.load(f)
                    currentSoilTension = loaded_variables['currentSoilTension']
                    self.currentPlot.threshold_timestamp = loaded_variables['threshold_timestamp']
                    self.currentPlot.predictions = loaded_variables['predictions']

                end_time = datetime.now().replace(microsecond=0)
                duration = end_time - start_time
                print("Prediction finished for plot: " + self.currentPlot.user_given_name + ", at: ", end_time, "The duration was: ", duration)

                # Call routine to irrigate
                if len(self.currentPlot.device_and_sensor_ids_flow) > 0: 
                    actuation.main(currentSoilTension, self.currentPlot.threshold_timestamp, self.currentPlot.predictions, self.currentPlot)

                # After initial training and prediction, start surveillance
                threading.Timer(10, self.currentPlot.check_threads).start()  # Check every hour if threads are alive

                # Wait for predict_period_hours periodically for next cycle
                time_to_sleep = self.time_until_n_hours(self.currentPlot.predict_period_hours)
                print(f"Waiting {time_to_sleep // 3600:.0f} hours {time_to_sleep % 3600 // 60:.0f} minutes until conducting next prediction...")
                time.sleep(time_to_sleep)  # Sleep until threshold
            except Exception as e:
                print(f"[{self.currentPlot.user_given_name }] Prediction thread error: {e}. Retrying after {Restart_time/60} minute.")
                # Release resources
                create_model.Currently_active = False
                time.sleep(Restart_time)  # Retry after 30 minute if there is an error

    def stop(self):
        self.stop_event.set()  # Signal the thread to stop

# Starts a prediction thread TODO: implement stop for old instances, check before
def start(currentPlot):
    # Do not start prediction if currently training 
    if not currentPlot.currently_training:
        # Stop previous prediction thread if it exists and is running
        if hasattr(currentPlot, 'prediction_thread') and currentPlot.prediction_thread is not None:
            if currentPlot.prediction_thread.is_alive():
                print("Stopping existing prediction thread...")
                currentPlot.prediction_thread.stop()
                currentPlot.prediction_thread.join()
        
        # Start prediction thread
        currentPlot.prediction_thread = PredictionThread(currentPlot, name="PredictionThread_" + str(currentPlot.user_given_name))
        currentPlot.prediction_thread.start()
    else:
        print("Perdiction Thread: Currently training, prediction will be started after training is finished.")