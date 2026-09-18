"""Stoppable schedule dispatch, independent of training and prediction cycles."""

import logging
import threading

import pandas as pd

import actuation
import plot_manager
from operations_store import get_operations_store

log = logging.getLogger(__name__)


class ScheduleDispatcher(threading.Thread):
    def __init__(self, *, store=None, get_plots=None, clock=None,
                 poll_seconds=30, batch_size=100):
        super().__init__(name='ScheduleDispatcher', daemon=True)
        self.store = store if store is not None else get_operations_store()
        self.get_plots = get_plots or plot_manager.getPlots
        self.clock = clock or (lambda: pd.Timestamp.now(tz='UTC'))
        self.started_at = self.clock()
        self.poll_seconds = poll_seconds
        self.batch_size = batch_size
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def _review(self, operation, reason):
        self.store.transition(operation['operation_id'], 'pending_approval',
                              {'actor': 'schedule_dispatcher', 'review_reason': reason})

    def _dispatch(self, operation):
        operation = self.store.get_operation(operation['operation_id'])
        if operation is None or operation['status'] not in {'planned', 'approved'}:
            return
        reason = actuation.schedule_window_error(operation, now=self.clock())
        if reason == 'schedule_not_due':
            return
        if reason:
            self.store.transition(operation['operation_id'], 'failed', {'error': reason})
            return
        start = pd.Timestamp(operation['planned_start'])
        start = start.tz_localize('UTC') if start.tzinfo is None else start.tz_convert('UTC')
        if operation.get('planned_end') is None and start < self.started_at:
            self._review(operation, 'missed_open_ended_schedule')
            return
        plots = list(self.get_plots().values())
        plot = next((plot for plot in plots if str(getattr(plot, 'stable_id', ''))
                     == str(operation['plot_id'])), None)
        if (plot is None or getattr(plot, 'farm_id', None) != operation['farm_id']
                or not actuation._has_actuator_support(plot)):
            self._review(operation, 'plot_or_actuator_unavailable')
            return
        mode = actuation.resolve_irrigation_mode(plot)
        if mode not in {'automatic', 'approval_required'} or (
                operation['status'] == 'planned' and
                (mode != 'automatic' or operation['mode'] != 'automatic')):
            self._review(operation, 'current_mode_requires_owner_review')
            return
        # Acknowledgement is not pump-stop feedback. Until shared-pump ownership
        # is implemented, do not autonomously dispatch ambiguous assignments.
        actuators = set(getattr(plot, 'device_and_sensor_ids_flow', []) or [])
        if any(other is not plot and actuators.intersection(
                getattr(other, 'device_and_sensor_ids_flow', []) or []) for other in plots):
            self._review(operation, 'shared_actuator_requires_owner_review')
            return
        if not self.stop_event.is_set():
            actuation.execute_operation_command(plot, operation)

    def run_once(self):
        cursor = 0
        now = self.clock()
        while not self.stop_event.is_set():
            page = self.store.due_schedules(now, after_rowid=cursor, limit=self.batch_size)
            if not page:
                break
            for cursor, operation in page:
                if self.stop_event.is_set():
                    return
                try:
                    self._dispatch(operation)
                except Exception:
                    # Claimed operations stay active after ambiguous failures;
                    # they are never selected for an automatic retry.
                    log.exception('Schedule dispatch failed for %s', operation['operation_id'])

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.run_once()
            except Exception:
                log.exception('Schedule scan failed')
            if self.stop_event.wait(self.poll_seconds):
                break
