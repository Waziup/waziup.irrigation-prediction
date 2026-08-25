import threading
from datetime import datetime, timedelta

# local
import runtime_config
import actuation
import prediction_thread
import create_model
import pipeline_cache


class TrainingThread(threading.Thread):
    def __init__(self, plot, start_immediately, name=None):
        super().__init__(name=name)
        self.daemon = True
        self.currentPlot = plot  # Attach process to a specific plot
        self.start_immediately = start_immediately
        self.stop_event = threading.Event()  # Stop flag

    def time_until_noon(self, train_period_days):
        """Calculate the time difference from now until the next noon."""
        now = datetime.now()
        noon_today = now.replace(hour=12, minute=0, second=0, microsecond=0)
        noon_today += timedelta(days=train_period_days)
        return (noon_today - now).total_seconds()

    def run_cycle(self):
        """Execute or restore one training-cycle result."""
        plot = self.currentPlot
        if create_model.state.Perform_training:
            # A real cycle supersedes any earlier cache-fallback state.
            plot._inference_source = "live"
            current_tension, threshold_timestamp, predictions = (
                create_model.run_training_cycle(plot)
            )
            pipeline_cache.save_cycle(
                plot,
                current_tension,
                threshold_timestamp,
                predictions,
            )
            plot.pipeline_cache_status = "saved"
            plot.pipeline_cache_reason = None
            return current_tension, threshold_timestamp, predictions

        # Reuse the same validated snapshot contract as prediction workers.
        payload, reason = pipeline_cache.load_cycle(plot.id)
        if reason is not None:
            pipeline_cache.record_unavailable(plot, reason)
            return None
        restored, reason = pipeline_cache.restore_cycle(
            plot,
            payload,
            max_age_hours=runtime_config.Max_fallback_prediction_age_hours,
        )
        if reason is not None:
            pipeline_cache.record_unavailable(plot, reason)
            return None
        return restored

    def run(self):
        # To stop via event
        while not self.stop_event.is_set():
            try:
                print(
                    f"Child process started for {self.currentPlot.user_given_name}")
                if not self.start_immediately:
                    # Wait until the next noon
                    time_to_sleep = self.time_until_noon(
                        runtime_config.get_timing_config(
                            self.currentPlot).retraining_interval_days)
                    print(
                        f"Waiting {time_to_sleep // 3600:.0f} hours {time_to_sleep % 3600 // 60:.0f} minutes until next training...")
                    if self.stop_event.wait(timeout=time_to_sleep):
                        break

                if self.stop_event.is_set():
                    break  # Exit if stopping

                self.currentPlot.currently_training = True
                start_time = datetime.now().replace(microsecond=0)
                print("Training for Plot:", self.currentPlot.user_given_name,
                      "started at:", start_time)

                cycle = self.run_cycle()
                if cycle is None:
                    self.currentPlot.training_finished = False
                    print(
                        f"[{self.currentPlot.user_given_name}] No usable cached training result "
                        f"({self.currentPlot.pipeline_cache_reason}); stopping training thread."
                    )
                    break
                currentSoilTension, self.currentPlot.threshold_timestamp, self.currentPlot.predictions = cycle

                self.currentPlot.training_finished = True
                self.currentPlot.currently_training = False
                # Only the user-triggered first cycle runs immediately.
                self.start_immediately = False

                # main() has returned: its train/val/test frames and scaled arrays are
                # gone with its stack frame, so this trim hands their pages back to the
                # OS before the days-long idle wait (no live objects are touched)
                create_model.free_memory(label="training thread idle")

                end_time = datetime.now().replace(microsecond=0)
                duration = end_time - start_time
                print("Training finished for plot: " + self.currentPlot.user_given_name +
                      ", at: ", end_time, "Duration:", duration)

                # Evaluate/persist alerts for every plot. Advisory-only plots
                # must receive the same stress-alert lifecycle as actuated plots.
                actuation.main(currentSoilTension, self.currentPlot.threshold_timestamp,
                               self.currentPlot.predictions, self.currentPlot)

                # Start prediction process if not running
                if (
                    (self.currentPlot.prediction_thread is None
                     or not self.currentPlot.prediction_thread.is_alive())
                    and not create_model.state.SkipDataPreprocessing
                    and not create_model.state.SkipTraining
                ):
                    prediction_thread.start(self.currentPlot)
                else:
                    print("Prediction thread is already running.")

            except Exception as e:
                self.currentPlot.currently_training = False
                self.currentPlot.training_finished = False
                self.currentPlot.pipeline_cache_status = "error"
                self.currentPlot.pipeline_cache_reason = str(e)
                print(
                    f"[{self.currentPlot.user_given_name}] Training thread error: {e}. "
                    f"Retrying after {runtime_config.get_timing_config(self.currentPlot).error_retry_seconds / 60} minutes.")
                if self.stop_event.wait(timeout=runtime_config.get_timing_config(
                        self.currentPlot).error_retry_seconds):
                    break

        self.currentPlot.currently_training = False

    def stop(self):
        self.stop_event.set()  # Signal the process to stop

# Starts a training process


def start(currentPlot):
    # Prediction must not enter a new model cycle while an explicit training
    # request is being prepared.
    if currentPlot.prediction_thread is not None and currentPlot.prediction_thread.is_alive():
        currentPlot.prediction_thread.stop()
        currentPlot.prediction_thread.join()

    # Stop previous training process
    if currentPlot.training_thread is not None:
        currentPlot.training_thread.stop()
        if currentPlot.training_thread.is_alive():
            currentPlot.training_thread.join()

    # Reset flags
    currentPlot.training_finished = False
    currentPlot.currently_training = True

    # Create and start a new training process
    currentPlot.training_thread = TrainingThread(
        currentPlot, True, name="TrainingThread_" + str(currentPlot.user_given_name))
    currentPlot.training_thread.start()
    print("Training thread started for plot:", currentPlot.user_given_name)
