import threading
import pathlib
import pickle
from datetime import datetime, timedelta, timezone

# local
import runtime_config
import actuation
import online_learning
import New_pipeline


class PredictionThread(threading.Thread):
    def __init__(self, plot, name=None):
        super().__init__(name=name)
        self.daemon = True
        self.currentPlot = plot  # Attach thread to a specific plot
        self.stop_event = threading.Event()  # Stop flag

    @staticmethod
    def time_until_n_hours(hours):
        """Return the number of seconds until the next prediction window."""
        now = datetime.now()
        predict_time = now + \
            timedelta(hours=hours, minutes=0, seconds=0, microseconds=0)

        return (predict_time - now).total_seconds()

    def run(self):
        # To stop via event
        while not self.stop_event.is_set():
            try:
                start_time = datetime.now().replace(microsecond=0)
                print(
                    f"Prediction cycle started for {self.currentPlot.user_given_name} at: {start_time}")

                file_path = pathlib.Path(
                    'data/cache/saved_variables_plot_' + str(self.currentPlot.id) + '.pkl')

                if create_model.state.Perform_training:  # same var is used here to preserve functionality
                    # Call predict_with_updated_data function
                    currentSoilTension, self.currentPlot.threshold_timestamp, self.currentPlot.predictions = create_model.predict_with_updated_data(
                        self.currentPlot)

                    # Create object to save
                    variables_to_save = {
                        'currentSoilTension': currentSoilTension,
                        'threshold_timestamp': self.currentPlot.threshold_timestamp,
                        'predictions': self.currentPlot.predictions,
                        'saved_at_utc': datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                    }
                    # Save the variables to a file
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(file_path, 'wb') as f:
                        pickle.dump(variables_to_save, f)
                else:
                    # Load the saved variables from the file
                    if not file_path.is_file():
                        print(
                            f"[{self.currentPlot.user_given_name}] Cached variables not found at {file_path}. "
                            "Cannot continue with Perform_training=False. Stopping prediction thread."
                        )
                        break

                    loaded_variables = None
                    try:
                        with open(file_path, 'rb') as f:
                            loaded_variables = pickle.load(f)
                    except (pickle.UnpicklingError, EOFError, KeyError, ValueError) as cache_error:
                        print(
                            f"[{self.currentPlot.user_given_name}] Cached variables are invalid ({cache_error}). "
                            "Falling back to manifest model."
                        )

                    max_age_h = float(getattr(
                        runtime_config, "Max_fallback_prediction_age_hours", 24))
                    cache_timestamp = None
                    if isinstance(loaded_variables, dict):
                        saved_at = loaded_variables.get('saved_at_utc')
                        if isinstance(saved_at, str):
                            try:
                                parsed = datetime.fromisoformat(
                                    saved_at.replace("Z", "+00:00"))
                                if parsed.tzinfo is not None:
                                    parsed = parsed.astimezone(
                                        timezone.utc).replace(tzinfo=None)
                                cache_timestamp = parsed
                            except ValueError:
                                cache_timestamp = None
                    if cache_timestamp is None and file_path.exists():
                        cache_timestamp = datetime.utcfromtimestamp(
                            file_path.stat().st_mtime)

                    cache_age_h = None
                    if cache_timestamp is not None:
                        cache_age_h = (
                            datetime.utcnow() - cache_timestamp
                        ).total_seconds() / 3600.0

                    use_cache = (
                        isinstance(loaded_variables, dict)
                        and cache_age_h is not None
                        and cache_age_h <= max_age_h
                    )

                    if use_cache:
                        currentSoilTension = loaded_variables['currentSoilTension']
                        self.currentPlot.threshold_timestamp = loaded_variables['threshold_timestamp']
                        self.currentPlot.predictions = loaded_variables['predictions']
                        self.currentPlot._loaded_model_timestamp = (
                            cache_timestamp.isoformat() + "Z" if cache_timestamp else None
                        )
                    else:
                        if cache_age_h is not None:
                            print(
                                f"[{self.currentPlot.user_given_name}] Cached predictions are stale "
                                f"(age={cache_age_h:.2f}h, max={max_age_h:.2f}h). "
                                "Falling back to manifest model."
                            )
                        try:
                            currentSoilTension, self.currentPlot.threshold_timestamp, self.currentPlot.predictions = (
                                New_pipeline._run_prediction_with_loaded_model(
                                    self.currentPlot)
                            )
                        except Exception as manifest_error:
                            print(
                                f"[{self.currentPlot.user_given_name}] Manifest prediction failed "
                                f"({manifest_error}). Stopping prediction thread."
                            )
                            break

                end_time = datetime.now().replace(microsecond=0)
                duration = end_time - start_time
                print("Prediction finished for plot: " + self.currentPlot.user_given_name +
                      ", at: ", end_time, "The duration was: ", duration)

                # Call routine to irrigate
                if len(self.currentPlot.device_and_sensor_ids_flow) > 0:
                    actuation.main(currentSoilTension, self.currentPlot.threshold_timestamp,
                                   self.currentPlot.predictions, self.currentPlot)

                # After initial training and prediction, start surveillance
                # Guard is per-cycle: avoids stacking timers during fast cycles.
                if not getattr(self.currentPlot, '_check_timer_pending', False):
                    self.currentPlot._check_timer_pending = True

                    def _guarded_check(plot=self.currentPlot):
                        plot._check_timer_pending = False
                        plot.check_threads()
                    threading.Timer(10, _guarded_check).start()

                # Wait for predict_period_hours periodically for next cycle
                time_to_sleep = self.time_until_n_hours(
                    self.currentPlot.predict_period_hours)
                print(
                    f"Waiting {time_to_sleep // 3600:.0f} hours {time_to_sleep % 3600 // 60:.0f} minutes until conducting next prediction...")
                if self.stop_event.wait(timeout=time_to_sleep):
                    break
            except Exception as e:
                print(
                    f"[{self.currentPlot.user_given_name}] Prediction thread error: {e}. Retrying after {Restart_time/60} minute.")
                # Release resources
                create_model.state.Currently_active = False
                # Retry after 30 minute if there is an error
                time.sleep(Restart_time)

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
        currentPlot.prediction_thread = PredictionThread(
            currentPlot, name="PredictionThread_" + str(currentPlot.user_given_name))
        currentPlot.prediction_thread.start()
        print("Prediction thread started for plot:",
              currentPlot.user_given_name)
    else:
        print("Prediction Thread: Currently training, prediction will be started after training is finished.")
