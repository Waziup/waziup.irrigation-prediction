(function () {
  "use strict";

  window.renderEOObservation = function (observation) {
    const ndvi = document.getElementById("eo-ndvi");
    const ndre = document.getElementById("eo-ndre");
    const trend = document.getElementById("eo-trend");
    const details = document.getElementById("eo-details");
    if (!ndvi || !ndre || !trend || !details) return;

    const validIndex = (value) => typeof value === "number"
      && Number.isFinite(value) && value >= -1 && value <= 1;
    const available = observation?.available === true;
    ndvi.textContent = available && validIndex(observation.latest_ndvi)
      ? observation.latest_ndvi.toFixed(2) : "--";
    ndre.textContent = available && validIndex(observation.latest_ndre)
      ? observation.latest_ndre.toFixed(2) : "--";

    const direction = observation?.trend?.direction;
    trend.textContent = available && ["increasing", "stable", "decreasing"].includes(direction)
      ? direction[0].toUpperCase() + direction.slice(1) : "--";

    if (!available) {
      details.textContent = observation?.reason || "EO observations are not available for this plot.";
      return;
    }

    const observed = new Date(observation.latest_observation);
    const date = Number.isNaN(observed.getTime())
      ? "Date unavailable" : observed.toLocaleDateString();
    // Provider details stay in the API; this summary shows when the scene was observed.
    details.textContent = `Latest observation: ${date}`;
  };
}());
