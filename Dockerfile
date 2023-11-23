FROM python:slim-buster

COPY . /root/src/

RUN  pip install requests

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
ENTRYPOINT ["python", "/root/src/main.py"]


# Here is how you can access inside your container:
# sudo docker exec -it waziup.hello-world-python sh