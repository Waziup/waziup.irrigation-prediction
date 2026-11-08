# WaziUps WaziApp: irrigation-prediction
This application (WaziApp) for the WaziGate predicts irrigation times with help of soil sensors and a weather API (open-meteo). A non-technical user guide with general instructions can be obtained from [here](help/user_guide.md).

## How to install

Visit the WaziGates UI and open the App section. Press the plus button and click install custom app. Type into the textbox "waziup/irrigation-prediction:latest" to download/install the application from dockerhub.

## How to change run configuration

For debugging there are 3 run configurations possible:

- **Production mode:** run as WaziApp in a docker container on the WaziGate
- **Local debugging environment:** Debug in Visual Studio with wazigate-edge and wazigate-dashboard running locally
- **Local debugging environment against local gateway:** Debug in Visual Studio with wazigate-edge and wazigate-dashboard running locally, using the API of a WaziGate that has its API exposed (e.g. local network).

Local debug settings use `API_URL` in [`.env`](.env). Production Compose uses
`WAZIGATE_API_URL` instead and defaults to `http://wazigate:80/`, so a local
`localhost` debug value cannot silently break the container. A remote gateway
also requires `WAZIGATE_USERNAME` and `WAZIGATE_PASSWORD`; do not commit either
credential. EO credentials must likewise be injected as deployment secrets.

## Persistent application state

Each WaziGate installation manages one farm with multiple plots. Farm details,
plot settings, sensor assignments, irrigation history, operations, events and
alerts are stored in the local SQLite file `data/wazifarm.sqlite3`. SQLite is
also the default when running Python directly. `APP_STATE_DB` can override the
path; Compose points it into the persistent WaziApp mount.

Farm records have no farm IDs or farm selection. Plot IDs remain stable across
restarts. A fresh installation has no farm or scaffold plot. The spatial storage
model requires a valid farm polygon and at least one valid plot polygon, saved
together. Plots must remain inside the farm and their interiors cannot overlap.
Areas are calculated by the backend from their polygons.


An empty installation opens `#/setup`; additional
plots use `#/setup/plot`. Existing metadata-only creation forms cannot create
spatial records.


## Local weather cache

The dashboard, model training, predictions, and crop calculations share hourly
weather in the application SQLite database. Recent weather refreshes after
30 minutes. Historical days fetched more than five days after the observation
date remain reusable without routine expiry, within the cache retention window.
Each successful provider segment is saved before the next request; retries fetch
only missing or expired blocks. Incomplete windows still fail explicitly, while
completed blocks remain available for the next attempt. The weather chart's
30-day browsing cap does not restrict weather required for model training.

## Restart recovery

With `APP_STATE_DB` configured, the application stores forecast snapshots in
`forecast_cache/` and complete fitted model state in `runtime_models/` alongside
the database. Keep that parent directory on the persistent application volume.
`IRRIGATION_CACHE_DIR` and `IRRIGATION_MODEL_STATE_DIR` can override these locations.

After a restart, compatible models resume prediction without immediate retraining.
Prediction scheduling rechecks wall-clock deadlines every 30 seconds, including
after host sleep. A periodic health check restarts stopped prediction workers
using their fitted models; it never initiates training for an untrained plot.
Recent cached forecasts are display-only and cannot replay irrigation commands.
Snapshots are authenticated and checked against plot configuration, implementation,
and dependency versions before reuse. Older model artifacts without preprocessing
state require one training cycle to create a recoverable snapshot.

## How to build

### Just clone the git and run the following cmd in the root folder:

    docker buildx build --platform linux/arm64/v8 -t waziup/irrigation-prediction:latest --no-cache --pull --build-arg CACHEBUST=$(date +%s) --load .

### Push to dockerhub:

Issue the following cmd to push to dockerhub (you have to be logged in):

    docker push waziup/irrigation-prediction:latest

### Copy image to local Raspberry Pi:

Create a folder with the name of the app at the following path: "/var/lib/wazigate/apps/waziup.irrigation-prediction" and copy the files (docker-compose.yml & package.json) from the repository.

Issue the following cmd to push to dockerhub (you have to be logged in):

    docker save {id of image} | gzip | pv | ssh pi@{ip of rpi} docker load

    # Example:
    docker save my-docker-image | gzip | pv | ssh pi@192.168.0.10 docker load
