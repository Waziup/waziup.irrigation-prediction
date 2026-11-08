"""Plot-scoped legacy volume-start and pump-stop controls."""
import math
import actuation
from operations_store import get_operations_store


def status(plot):
    store = get_operations_store()
    latest = store.latest_for_plot(plot.stable_id)
    return {'operation_id': latest['operation_id'] if latest else None,
            'operation_status': latest['status'] if latest else None,
            'plot_id': plot.stable_id,
            'configured': actuation._has_actuator_support(plot),
            'mode': actuation.resolve_irrigation_mode(plot),
            'busy': bool(store.latest_for_plot(plot.stable_id, statuses={'active'})
                         or store.has_pending_flow_verification(plot.stable_id))}


def command(plot, state, amount=None, key=None):
    store = get_operations_store()
    if not actuation._has_actuator_support(plot):
        return 409, {'message': 'Select an irrigation actuator in Plot settings.'}
    if state == 'off':
        if not actuation.set_pump_state(plot, False):
            return 502, {'message': 'The OFF command could not be confirmed. Check the controller.'}
        # Cancel pending verification without treating the requested volume as delivered.
        operations = store.list_operations(plot_id=plot.stable_id, status='completed', limit=500)
        active = store.latest_for_plot(plot.stable_id, statuses={'active'})
        if active:
            operations.append(active)
        for operation in operations:
            if operation['status'] not in {'active', 'completed'}:
                continue
            context = (operation.get('recommendation') or {}).get('flow_control')
            if context is None:
                for event in reversed(store.events(operation['operation_id'])):
                    context = (event.get('detail') or {}).get('flow_verification')
                    if context:
                        break
            if operation['status'] == 'completed' and not context:
                continue
            detail = {'command': 'pump_off', 'reason': 'farmer_stopped_irrigation'}
            destination = 'failed'
            if context:
                try:
                    delivered, evidence = actuation._measured_delivery(plot, context)
                    store.update_operation(operation['operation_id'], amount_m3=delivered)
                    detail.update(evidence)
                    detail['delivered_m3'] = delivered
                    destination = 'verified'
                except Exception:
                    detail['error'] = 'delivery_unverified_after_manual_stop'
            try:
                store.transition(operation['operation_id'], destination, detail)
            except ValueError:
                pass  # A meter-monitor completion may have won the transition.
        return 200, {'plot_id': plot.stable_id, 'status': 'sent', 'message': 'OFF command sent.'}
    if state != 'on':
        return 400, {'message': 'Choose ON or OFF.'}
    if actuation.resolve_irrigation_mode(plot) != 'manual':
        return 409, {'message': 'Select Semi-Auto in Plot settings to start irrigation yourself.'}
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return 400, {'message': 'Enter a water amount in m³.'}
    if not math.isfinite(amount) or not 0.01 <= amount <= 1000:
        return 400, {'message': 'Enter an amount between 0.01 and 1000 m³.'}
    if not isinstance(key, str) or not key.strip() or len(key) > 160:
        return 400, {'message': 'An irrigation request ID is required.'}
    operation, created = store.create_operation(
        idempotency_key=f'volume:{plot.stable_id}:{key}', plot_id=plot.stable_id,
        plot_name=plot.user_given_name, source='manual', mode='manual',
        status='approved', amount_m3=amount)
    if not created:
        if operation.get('amount_m3') != amount:
            return 409, {'message': 'This request ID was already used for another amount.'}
        return 200, {'plot_id': plot.stable_id, 'status': 'duplicate', 'operation_id': operation['operation_id'],
                     'message': 'This irrigation request was already submitted.'}
    if not actuation.execute_operation_command(plot, operation):
        return 409, {'message': 'Irrigation could not start. Check the controller and any pending delivery confirmation.'}
    return 200, {'plot_id': plot.stable_id, 'status': 'sent', 'operation_id': operation['operation_id'],
                 'message': f'ON command sent for {amount:g} m³.'}
