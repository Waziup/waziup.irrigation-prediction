"""Separate soil measurement units without guessing from numeric readings."""


def soil_sensor_groups(plot, sensor_ids):
    """Return tension, VWC and ambiguous IDs from configured soil sensors.

    CSV names identify their measurement type. Opaque gateway IDs retain the
    configured single-type plot contract; a mixed plot needs identifiable
    types and must not silently treat unknown IDs as tension.
    """
    mode = str(getattr(plot, "sensor_kind", "tension") or "tension").lower()
    tension, vwc, unknown = [], [], []
    for sensor_id in dict.fromkeys(sensor_ids or []):
        name = str(sensor_id).rsplit("/", 1)[-1].lower()
        if name.startswith(("vwc", "volumetric", "capacitive")):
            vwc.append(sensor_id)
        elif name.startswith("tension") and mode != "capacitive":
            tension.append(sensor_id)
        elif mode == "tension":
            tension.append(sensor_id)
        elif mode == "capacitive":
            vwc.append(sensor_id)
        else:
            unknown.append(sensor_id)
    return tension, vwc, unknown


def update_soil_sensor_groups(plot):
    groups = soil_sensor_groups(plot, plot.device_and_sensor_ids_moisture)
    (plot.device_and_sensor_ids_tension, plot.device_and_sensor_ids_vwc,
     plot.device_and_sensor_ids_unknown) = groups
    return groups
