(function () {
  "use strict";

  const active = new URLSearchParams(window.location.search).get("demo") === "true";
  let selectedPlotId = "demo-plot-maize";
  const plotProfiles = {
    "demo-plot-maize": {name: "North Maize Field", crop: "Maize", stage: "Mid-season", tension: 38.4, temperature: 24.6, vwc: 27.8, threshold: 35, urgency: "critical", nextAction: "Irrigate today", margin: -3.4, volume: 8.4, rain: 0.4, kc: 1.16, gdd: 912, latitude: -0.102610, longitude: 34.761710, area: 2400, mode: "automatic", ndvi: 0.71, ndre: 0.34, eoTrend: "increasing"},
    "demo-plot-tomato": {name: "Greenhouse Tomatoes", crop: "Tomato", stage: "Development", tension: 25.1, temperature: 23.2, vwc: 32.5, threshold: 32, urgency: "watch", nextAction: "Monitor; irrigation likely tomorrow", margin: 6.9, volume: 3.15, rain: 1.2, kc: 0.82, gdd: 541, latitude: -0.103120, longitude: 34.762340, area: 900, mode: "approval_required", ndvi: 0.62, ndre: 0.28, eoTrend: "stable"},
    "demo-plot-beans": {name: "Lower Bean Plot", crop: "Beans", stage: "Late-season", tension: 19.7, temperature: 22.8, vwc: 36.1, threshold: 34, urgency: "ok", nextAction: "No irrigation needed today", margin: 14.3, volume: 0, rain: 4.8, kc: 0.71, gdd: 1088, latitude: -0.104020, longitude: 34.760910, area: 1700, mode: "manual", ndvi: 0.54, ndre: 0.23, eoTrend: "decreasing"},
  };
  const plotIds = Object.keys(plotProfiles);
  const profile = () => plotProfiles[selectedPlotId];
  const isoAt = (hours) => new Date(Date.now() + hours * 3600000).toISOString();
  const dateAt = (days) => { const value = new Date(); value.setDate(value.getDate() + days); return value.toISOString().slice(0, 10); };
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const plotRecords = () => plotIds.map((id, index) => ({plot_id: id, farm_id: "demo-farm", name: plotProfiles[id].name, area: plotProfiles[id].area, area_unit: "m2", ordinal: index + 1}));

  function farmDashboard() {
    return {schema_version: "1.0", generated_at: new Date().toISOString(),
      farm: {farm_id: "demo-farm", name: "Sunrise Demo Farm", size: 5000, area_unit: "m2", timezone: "Africa/Nairobi"},
      summary: {total_plots: 3, plots_needing_attention: 2, planned_water_m3: 11.55},
      plots: plotIds.map((id) => { const p = plotProfiles[id]; return {plot_id: id, farm_id: "demo-farm", name: p.name, crop_type: p.crop, growth_stage: p.stage, current_tension_cbar: p.tension, threshold_cbar: p.threshold, threshold_margin_cbar: p.margin, next_action: p.nextAction, urgency: p.urgency, next_action_time: p.volume ? isoAt(6) : null, data_state: "fresh"}; }),
      active_alerts: [{alert_id: "demo-alert-1", plot_id: "demo-plot-maize", urgency: "critical"}, {alert_id: "demo-alert-2", plot_id: "demo-plot-tomato", urgency: "watch"}],
      todays_plan: [{plot_id: "demo-plot-maize", status: "planned", amount_m3: 8.4}, {plot_id: "demo-plot-tomato", status: "planned", amount_m3: 3.15}]};
  }

  function chartPayload(forecast) {
    const p = profile(); const count = forecast ? 12 : 18; const start = forecast ? 1 : -17;
    return {available: true, kind: "tension", unit: "cbar", timestamps: Array.from({length: count}, (_, i) => isoAt(start + i)),
      moistureSeries: Array.from({length: count}, (_, i) => Number((p.tension + (forecast ? i * 0.55 : (i - count + 1) * 0.32)).toFixed(1))),
      vwcSeries: Array.from({length: count}, (_, i) => Number((p.vwc - (forecast ? i * 0.18 : (i - count + 1) * 0.08)).toFixed(1))),
      threshold_mode: "dynamic", threshold_cbar: p.threshold, saturation: 0, fieldCapacityLower: 10, fieldCapacityUpper: 22, permanentWiltingPoint: 55};
  }

  function recommendation() {
    const p = profile(); const needed = p.volume > 0;
    return {available: true,
      action: {should_irrigate: needed, advisory_only: true, mode: p.mode, urgency: p.urgency, label: p.nextAction},
      condition: {current_tension_cbar: p.tension, threshold_cbar: p.threshold, margin_cbar: p.margin, status: p.margin <= 0 ? "threshold_exceeded" : "below_threshold", message: needed ? "The field is approaching its irrigation trigger." : "Rain and soil moisture are sufficient."},
      crop: {type: p.crop.toLowerCase(), crop_type: p.crop, crop_name: p.crop, current_stage: p.stage, gdd_cumulative: p.gdd, kc: p.kc},
      water: {recommended_volume_m3: p.volume, calculated_requirement_volume_m3: p.volume, applied_since_previous_calculation_m3: 0, outlook: {requirement: {remaining_net_irrigation_mm: needed ? 2.8 : 0, remaining_gross_irrigation_mm: needed ? 3.5 : 0}}},
      weather: {etc_daily_mm: 4.2, rain_mm: p.rain, age_hours: 1.2}, satellite: {ndvi: p.ndvi, ndre: p.ndre, source_age_hours: 18}, freshness: {stale: false, stale_sources: []}, timing: {first_breach_timestamp: needed ? isoAt(6) : null}};
  }

  function phenology() {
    const p = profile();
    return {available: true, crop_type: p.crop, crop_name: p.crop, current_stage: p.stage, threshold_mode: "dynamic", threshold_active_cbar: p.threshold, planting_date: dateAt(-68), evaluated_at: new Date().toISOString(),
      stage_rows: [{stage: "Pre-emergence", gdd_start: 0, gdd_end: 120, kc_start: 0.3, kc_end: 0.4, threshold_cbar: 24}, {stage: "Development", gdd_start: 120, gdd_end: 600, kc_start: 0.4, kc_end: 0.85, threshold_cbar: 30}, {stage: "Mid-season", gdd_start: 600, gdd_end: 1050, kc_start: 0.85, kc_end: 1.16, threshold_cbar: p.threshold}, {stage: "Late-season", gdd_start: 1050, gdd_end: 1450, kc_start: 1.16, kc_end: 0.65, threshold_cbar: 38}]};
  }

  function fixture(path) {
    const pathname = new URL(path, window.location.href).pathname; const p = profile();
    return ({
      "/api/getPlots": {tabnames: plotIds.map((id) => plotProfiles[id].name), currentPlot: plotIds.indexOf(selectedPlotId) + 1, current_plot_id: selectedPlotId, current_farm_id: "demo-farm", plots: plotRecords()},
      "/api/getFarmRegistry": {current_farm_id: "demo-farm", current_plot_id: selectedPlotId, farms: [{farm_id: "demo-farm", name: "Sunrise Demo Farm", size: 5000, area_unit: "m2", timezone: "Africa/Nairobi", plot_ids: plotIds}], plots: plotRecords()},
      "/api/returnConfig": {data: {Name: p.name, Crop_type: p.crop, Plot_area_m2: p.area, Plot_area_unit: "m2", Gps_info: {latitude: p.latitude, longitude: p.longitude}}},
      "/api/getAlertStatus": p.urgency === "ok" ? {available: false} : {available: true, urgency: p.urgency},
      "/api/farmDashboard": farmDashboard(),
      "/api/getValuesForDashboard": {available: true, moisture_average: p.tension, temp_average: p.temperature, vwc_average: p.vwc, sensor_kind: "both", data_source: "Demo sensor", recorded_or_live: "simulated"},
      "/api/checkActiveIrrigation": {data: {activeIrrigation: false, actuatorConfigured: false, mode: "advisory_only"}},
      "/api/getHistoricalChartData": chartPayload(false), "/api/getPredictionChartData": chartPayload(true),
      "/api/getThreshold": p.volume ? {threshold: true, timestamp: isoAt(6)} : {threshold: false},
      "/api/getIrrigationRecommendation": recommendation(), "/api/getPhenologySummary": phenology(),
      "/api/getEOObservation": {available: true, latest_ndvi: p.ndvi, latest_ndre: p.ndre, latest_observation: isoAt(-18), valid_observations: 8, trend: {direction: p.eoTrend}},
      "/api/getWeatherForecast": {days: Array.from({length: 5}, (_, i) => ({date: dateAt(i), rain_mm: [0.4, 1.2, 4.8, 0, 2.1][i], temperature_min_c: 17 + i % 2, temperature_max_c: 27 + i % 3, humidity_percent: 64 + i * 2, wind_speed: 7 + i, icon: ["sun", "cloud", "rain", "sun", "cloud"][i]}))},
    })[pathname];
  }

  const response = (payload, status = 200) => new Response(JSON.stringify(payload), {status, headers: {"Content-Type": "application/json"}});
  window.WaziFarmDemo = {active, get selectedPlotId() { return selectedPlotId; }, async fetch(path, options) {
    const method = String(options?.method || "GET").toUpperCase(); const pathname = new URL(path, window.location.href).pathname;
    if (method === "POST" && pathname === "/api/setPlot") { const requested = new URLSearchParams(options?.body || "").get("plot_id"); if (!plotProfiles[requested]) return response({error: "Unknown demo plot"}, 404); selectedPlotId = requested; return response({status: "ok", plot_id: requested, currentPlot: plotIds.indexOf(requested) + 1}); }
    if (method !== "GET") return response({error: "Demo mode is read-only"}, 405);
    const payload = fixture(path); return payload === undefined ? response({error: "No demo fixture for this request"}, 404) : response(clone(payload));
  }};
}());
