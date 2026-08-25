import threading
from datetime import datetime, timedelta

# local
import runtime_config
import actuation
import create_model
import pipeline_cache


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

    def run_cycle(self):
        """Execute or restore one prediction cycle."""
        plot = self.currentPlot
        if create_model.state.Perform_training:
            # A real cycle supersedes any earlier cache-fallback state.
            plot._inference_source = "live"
            current_tension, threshold_timestamp, predictions = (
                create_model.run_prediction_cycle(plot)
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

        # Training-disabled deployments may display only a recent valid cache.
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
                start_time = datetime.now().replace(microsecond=0)
                print(
                    f"Prediction cycle started for {self.currentPlot.user_given_name} at: {start_time}")

                cycle = self.run_cycle()
                if cycle is None:
                    print(
                        f"[{self.currentPlot.user_given_name}] No usable cached prediction "
                        f"({self.currentPlot.pipeline_cache_reason}); stopping prediction thread."
                    )
                    break
                currentSoilTension, self.currentPlot.threshold_timestamp, self.currentPlot.predictions = cycle

                end_time = datetime.now().replace(microsecond=0)
                duration = end_time - start_time
                print("Prediction finished for plot: " + self.currentPlot.user_given_name +
                      ", at: ", end_time, "The duration was: ", duration)

                # Evaluate/persist alerts for every plot. actuation.main itself
                # decides whether the configured mode permits a hardware command.
                actuation.main(currentSoilTension, self.currentPlot.threshold_timestamp,
                               self.currentPlot.predictions, self.currentPlot)

                # After initial training and prediction, start surveillance
                # Guard is per-cycle: avoids stacking timers during fast cycles.
                if not getattr(self.currentPlot, '_check_timer_pending', False):
                    self.currentPlot._check_timer_pending = True

                    def _guarded_check(plot=self.currentPlot):
                        plot._check_timer_pending = False
                        plot.check_threads()
                    timer = threading.Timer(10, _guarded_check)
                    # This health-check timer must not keep the process alive.
                    timer.daemon = True
                    self.currentPlot._check_thread_timer = timer
                    timer.start()

                # Wait for predict_period_hours periodically for next cycle
                prediction_interval_hours = runtime_config.get_timing_config(
                    self.currentPlot).prediction_interval_hours
                time_to_sleep = self.time_until_n_hours(
                    prediction_interval_hours)
                print(
                    f"Waiting {time_to_sleep // 3600:.0f} hours {time_to_sleep % 3600 // 60:.0f} minutes until conducting next prediction...")
                if self.stop_event.wait(timeout=time_to_sleep):
                    break
            except Exception as e:
                self.currentPlot.pipeline_cache_status = "error"
                self.currentPlot.pipeline_cache_reason = str(e)
                print(
                    f"[{self.currentPlot.user_given_name}] Prediction thread error: {e}. "
                    f"Retrying after {runtime_config.get_timing_config(self.currentPlot).error_retry_seconds / 60} minutes.")
                if self.stop_event.wait(timeout=runtime_config.get_timing_config(
                        self.currentPlot).error_retry_seconds):
                    break

    def stop(self):
        self.stop_event.set()  # Signal the thread to stop
        # Prevent a delayed health check from running after worker shutdown.
        timer = getattr(self.currentPlot, "_check_thread_timer", None)
        if timer is not None:
            timer.cancel()
            self.currentPlot._check_timer_pending = False

# Start one prediction worker per plot.


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
