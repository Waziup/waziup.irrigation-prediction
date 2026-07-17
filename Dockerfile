# ---------- Builder: compile/install all Python deps, then discard the toolchain ----------
FROM python:3.9-slim-bullseye AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Build-only toolchain — NOT carried into the final image
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc build-essential pkg-config libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install into /root/.local so the whole tree can be copied to the runtime stage.
# --no-cache-dir keeps pip's wheel cache (hundreds of MB for the TF wheel) out of the layer.
RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --user --no-cache-dir --retries 10 --timeout 120 tensorflow-aarch64 && \
    pip install --user --no-cache-dir --retries 10 --timeout 120 pycaret && \
    pip install --user --no-cache-dir --retries 10 --timeout 120 \
        scikit-learn xgboost catboost scikeras matplotlib keras-tuner && \
    pip install --user --no-cache-dir --retries 10 --timeout 120 \
        requests==2.28.2 "urllib3<2.0" requests-unixsocket==0.2.0 pytz \
        timezonefinder python-dotenv python-dateutil joblib==1.3 psutil gevent
# xmlrunner dropped (test-only; the previous comment noted it isn't needed in the image).
# Add it back on the line above if you run unittests inside the container.

# Drop bytecode caches from the copied tree
RUN find /root/.local -name '__pycache__' -type d -prune -exec rm -rf {} + || true


# ---------- Runtime: slim image with only what's needed to RUN ----------
FROM python:3.9-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive
# Cap glibc malloc arenas (4GB RPi)
ENV MALLOC_ARENA_MAX=2
# Make the --user installed packages importable
ENV PATH=/root/.local/bin:$PATH

# Runtime-only OS deps:
#   curl        -> weather API is fetched via `subprocess ['curl', url]` in create_model/weather.py
#   expect      -> provides `unbuffer`, used in the ENTRYPOINT
#   zip         -> builds /index.zip below (WaziApp packaging artifact)
#   libhdf5-103 -> h5py runtime shared lib (previously pulled in by libhdf5-dev)
#   libgomp1    -> OpenMP runtime for xgboost/catboost (previously came free with build-essential)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl expect zip libhdf5-103 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installed Python packages from the builder
COPY --from=builder /root/.local /root/.local

# App source + WaziApp packaging
COPY . /root/src/
RUN mkdir -p /root/src/data \
    && cd /root/src/ \
    && mkdir -p /var/lib/waziapp \
    && zip /index.zip docker-compose.yml package.json

WORKDIR /root/src/

# Uncomment For development
#ENTRYPOINT ["tail", "-f", "/dev/null"]

# Uncomment For production
ENTRYPOINT ["sh", "-c", "unbuffer python main.py 2>&1 | tee -a python_logs.log"]


# Here is how you can access inside your container:
# sudo docker exec -it waziup.irrigation-prediction bash
