"""Shared read policy for real watering operations, including legacy records."""
import math


def is_irrigation_operation(operation):
    if (operation.get('source') == 'recommendation' and
            (operation.get('recommendation') or {}).get('superseded')):
        return False
    # Never hide actual delivery/history, including a later failed delivery.
    if operation.get('source') != 'recommendation' or operation.get('status') not in {
            'planned', 'pending_approval', 'approved'}:
        return True
    decision = operation.get('recommendation') or {}
    try:
        amount = float(operation.get('amount_m3'))
    except (TypeError, ValueError):
        return False
    return (decision.get('should_irrigate') is True and math.isfinite(amount)
            and amount > 0 and bool(operation.get('planned_start')))


def irrigation_plan(operations):
    """Filter display-only false plans without deleting the audit records."""
    return [item for item in operations if is_irrigation_operation(item)]


def planned_water_total(operations):
    # Delivered water belongs to history; failed/declined water is not planned.
    return round(sum(float(item.get('amount_m3') or 0) for item in operations
                     if item.get('status') in {'planned', 'pending_approval', 'approved'}), 3)
