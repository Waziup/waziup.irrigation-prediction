FROM python:3.7-slim

COPY . /root/src/

RUN apt-get update \
    && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN  pip install requests \
     pycaret \
     matplotlib \
     pytz \
     requests \
     geopy \
     timezonefinder
     
RUN  apt-get update \
     && apt-get install -y \
     curl \
     zip \
     && cd /root/src/ \
     && mkdir -p /var/lib/waziapp \
     && zip /index.zip docker-compose.yml package.json

#----------------------------#

# Uncomment For development
#ENTRYPOINT ["tail", "-f", "/dev/null"]


# Uncomment For production
WORKDIR /root/src/
ENTRYPOINT ["sh", "-c", "python main.py > logs.log"]


# Here is how you can access inside your container:
# sudo docker exec -it waziup.irrigation-prediction bash