FROM python:3.7-slim 
#later alpine to save even more, but is it worth it?

RUN apt-get update \
    && apt-get install -y \
    gcc \
    curl \
    zip \
    expect

RUN  pip install requests \
     pycaret[regression] \
     matplotlib \
     pytz \
     requests \
     geopy \
     timezonefinder \
     python-dotenv

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