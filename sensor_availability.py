"""Limit plot sensor discovery to unassigned and currently selected channels."""

SENSOR_FIELDS = (
    'device_and_sensor_ids_moisture',
    'device_and_sensor_ids_temp',
    'device_and_sensor_ids_flow_confirmation',
)


def available_plot_devices(devices, current_plot, plots):
    def selected(plot):
        return {str(channel) for field in SENSOR_FIELDS
                for channel in (getattr(plot, field, None) or [])}

    own = selected(current_plot)
    reserved = set()
    for plot in plots:
        if plot is not current_plot and plot.stable_id != current_plot.stable_id:
            reserved.update(selected(plot))
    result = []
    for device in devices:
        sensors = []
        for sensor in device.get('sensors', []):
            identities = {f"{device['id']}/{sensor['id']}"}
            if sensor.get('data_source'):
                identities.add(str(sensor['data_source']))
            if identities & own or not identities & reserved:
                sensors.append(sensor)
        # Preserve actuators and device metadata; only sensor choices change.
        result.append({**device, 'sensors': sensors})
    return result
