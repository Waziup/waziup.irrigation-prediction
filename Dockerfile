FROM python:3.8-slim
#later alpine to save even more, but is it worth it?

# Set environment variables to ensure non-interactive apt-get and prevent cache busting
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    zip \
    expect \
    build-essential \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN  pip install requests \
     pycaret \
     matplotlib \
     pytz \
     requests \
     geopy \
     timezonefinder \
     python-dotenv \
     xgboost \
     catboost \
     tensorflow \
     scikeras \
     scikit-learn \
     joblib==1.3 \
     keras-tuner
# keras tuner check usage!
COPY . /root/src/

RUN rm -rf /var/lib/apt/lists/* \
    && cd /root/src/ \
    && mkdir -p /var/lib/waziapp \
    && zip /index.zip docker-compose.yml package.json
     
#----------------------------#

# Uncomment For development
#ENTRYPOINT ["tail", "-f", "/dev/null"]


# Uncomment For production
WORKDIR /root/src/
ENTRYPOINT ["sh", "-c", "unbuffer python main.py 2>&1 | tee -a python_logs.log"]


# Here is how you can access inside your container:
# sudo docker exec -it waziup.irrigation-prediction bash