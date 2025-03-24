import multiprocessing
import time
import pathlib
import pickle
from datetime import datetime, timedelta

# local
import create_model
import actuation
import prediction_thread
import plot_manager

Restart_time = 1800  # DEBUG 1800 ~ 30 min in s

class TrainingProcess(multiprocessing.Process):
    def __init__(self, plot, startTrainingNow):
        super().__init__()
        self.currentPlot = plot  # Attach process to a specific plot
        self.startTrainingNow = startTrainingNow
        self.stop_event = multiprocessing.Event()  # Stop flag

    def time_until_noon(self, train_period_days):
        """Calculate the time difference from now until the next noon."""
        now = datetime.now()
        noon_today = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if now >= noon_today:
            noon_today += timedelta(days=train_period_days)
        return (noon_today - now).total_seconds()
    
    # Since with multiprocessing we can't pass objects, we need to write the plot back to the manager
    def write_plot_to_manager(self):
        try:
            if self.currentPlot.tab_number == plot_manager.Plots[self.currentPlot.tab_number].tab_number:
                plot_manager.Plots[self.currentPlot.tab_number].predictions = self.currentPlot.predictions
                plot_manager.Plots[self.currentPlot.tab_number].threshold_timestamp= self.currentPlot.threshold_timestamp
                plot_manager.Plots[self.currentPlot.tab_number].training_finished = self.currentPlot.training_finished
                plot_manager.Plots[self.currentPlot.tab_number].currently_training = self.currentPlot.currently_training
            else:
                for plot in plot_manager.Plots: #TODO: not ready yet
                    if plot.id == self.currentPlot.id:
                        plot_manager.Plots[plot.tab_number] = self.currentPlot
                        break
        except Exception as e:
            print(f"Error updating plot in manager: {e}")

    def run(self):
        # To stop via event
        while not self.stop_event.is_set():
            try:
                #with create_model.Create_model_lock:  # Acquire the lock
                print(f"Child process started for {self.currentPlot.user_given_name}")
                if not self.startTrainingNow:
                    # Wait until the next noon
                    time_to_sleep = self.time_until_noon(self.currentPlot.train_period_days)
                    print(f"Waiting {time_to_sleep // 3600:.0f} hours {time_to_sleep % 3600 // 60:.0f} minutes until next training...")
                    time.sleep(time_to_sleep)

                if self.stop_event.is_set():
                    break  # Exit if stopping

                start_time = datetime.now().replace(microsecond=0)
                print("Training for Plot:", self.currentPlot.user_given_name, "started at:", start_time)

                file_path = pathlib.Path('saved_variables_plot_' + str(self.currentPlot.id) + '.pkl')

                if create_model.Perform_training:
                    # Call create model function
                    currentSoilTension, self.currentPlot.threshold_timestamp, self.currentPlot.predictions = create_model.main(self.currentPlot)

                    # Save variables to a file
                    variables_to_save = {
                        'currentSoilTension': currentSoilTension,
                        'threshold_timestamp': self.currentPlot.threshold_timestamp,
                        'predictions': self.currentPlot.predictions
                    }
                    with open(file_path, 'wb') as f:
                        pickle.dump(variables_to_save, f)

                    # Write back the plot to the plot manager
                    self.write_plot_to_manager()
                else:
                    # Load saved variables
                    with open(file_path, 'rb') as f:
                        loaded_variables = pickle.load(f)
                    currentSoilTension = loaded_variables['currentSoilTension']
                    self.currentPlot.threshold_timestamp = loaded_variables['threshold_timestamp']
                    self.currentPlot.predictions = loaded_variables['predictions']

                self.currentPlot.training_finished = True
                self.currentPlot.currently_training = False
                self.startTrainingNow = False

                end_time = datetime.now().replace(microsecond=0)
                duration = end_time - start_time
                print("Training finished for plot: " + self.currentPlot.user_given_name + ", at: ", end_time, "Duration:", duration)

                # Call routine to irrigate
                if len(self.currentPlot.device_and_sensor_ids_flow) > 0:
                    actuation.main(currentSoilTension, self.currentPlot.threshold_timestamp, self.currentPlot.predictions, self.currentPlot)

                # # Start prediction process if not running
                # if self.currentPlot.prediction_thread is None:
                #     prediction_thread.start(self.currentPlot)

            except Exception as e:
                print(f"Training error: {e}. Retrying after 30 minutes.")
                time.sleep(Restart_time)

    def stop(self):
        self.stop_event.set()  # Signal the process to stop

# Starts a training process
def start(currentPlot):
    # Stop previous training process
    if currentPlot.training_process is not None:
        currentPlot.training_process.stop()
        currentPlot.training_process.join()

    # Reset flags
    currentPlot.training_finished = False
    currentPlot.currently_training = True

    # Create and start a new training process
    currentPlot.training_process = TrainingProcess(currentPlot, True)
    currentPlot.training_process.start()