"""Pure farm-dashboard composition from stable application contracts."""

from datetime import datetime, timezone


def _operation_for_plot(operations, plot_id):
    matches = [item for item in operations if item.get("plot_id") == plot_id]
    return matches[0] if matches else None


def build_farm_dashboard(*, farm, plot_records, runtime_plots, recommendations,
                         alerts, operations, generated_at=None):
    generated_at = generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    alert_by_plot = {item["plot_id"]: item for item in alerts}
    cards = []
    equipment_configured = 0
    equipment_active = 0
    stale_count = 0
    for record in plot_records:
        plot_id = record["plot_id"]
        runtime = runtime_plots.get(plot_id)
        recommendation = recommendations.get(plot_id) or {}
        alert = alert_by_plot.get(plot_id)
        operation = _operation_for_plot(operations, plot_id)
        condition = recommendation.get("condition") or {}
        crop = recommendation.get("crop") or {}
        action = recommendation.get("action") or {}
        timing = recommendation.get("timing") or {}
        freshness = recommendation.get("freshness") or {}
        flow_ids = getattr(runtime, "device_and_sensor_ids_flow", []) if runtime else []
        configured = bool(flow_ids)
        equipment_configured += int(configured)
        if operation and operation.get("status") == "active":
            equipment_state = "active"; equipment_active += 1
        elif not configured:
            equipment_state = "not_configured"
        elif action.get("mode") == "advisory_only":
            equipment_state = "advisory_only"
        else:
            equipment_state = "ready"
        data_state = "unavailable" if not recommendation.get("available") else (
            "stale" if freshness.get("stale") else "fresh")
        stale_count += int(data_state != "fresh")
        cards.append({
            "plot_id": plot_id, "farm_id": record.get("farm_id"),
            "name": record.get("name") or getattr(runtime, "user_given_name", ""),
            "crop_type": crop.get("type") or getattr(runtime, "crop_type", None),
            "growth_stage": crop.get("current_stage"),
            "current_tension_cbar": condition.get("current_tension_cbar"),
            "threshold_cbar": condition.get("threshold_cbar"),
            "threshold_margin_cbar": condition.get("margin_cbar"),
            "condition_status": condition.get("status", "unknown"),
            "next_action": action.get("label", "Awaiting forecast"),
            "urgency": action.get("urgency", "unknown"),
            "next_action_time": timing.get("first_breach_timestamp") or timing.get("first_breach_horizon"),
            "alert": ({"alert_id": alert.get("alert_id"), "urgency": alert.get("urgency"),
                       "payload": alert.get("payload")} if alert else None),
            "equipment": {"configured": configured, "state": equipment_state,
                          "mode": action.get("mode")},
            "operation": ({"operation_id": operation.get("operation_id"),
                           "status": operation.get("status"),
                           "amount_m3": operation.get("amount_m3")} if operation else None),
            "data_state": data_state,
        })

    urgency_counts = {name: sum(1 for alert in alerts if alert.get("urgency") == name)
                      for name in ("watch", "advise", "critical")}
    included = [item for item in operations if item.get("status") not in {"declined", "failed"}]
    planned_water = round(sum(float(item.get("amount_m3") or 0) for item in included), 3)
    return {
        "schema_version": "1.0", "generated_at": generated_at,
        "farm": {"farm_id": farm.get("farm_id"), "name": farm.get("name"),
                 "size": farm.get("size"), "area_unit": farm.get("area_unit"),
                 "timezone": farm.get("timezone")},
        "summary": {"total_plots": len(cards), "plots_needing_attention": len(alerts),
                    "alert_counts": urgency_counts, "planned_water_m3": planned_water,
                    "equipment_configured": equipment_configured,
                    "equipment_active": equipment_active,
                    "plots_with_stale_or_unavailable_data": stale_count},
        "plots": cards, "active_alerts": alerts, "todays_plan": operations,
    }
