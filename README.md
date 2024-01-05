# WaziUps WaziApp: irrigation-prediction
This application (WaziApp) for the WaziGate predicts irrigation times with help of soil sensors and a weather API (open-meteo).

## How to build 
### Just clone the git and run the following cmd in the root folder:
    docker buildx build --platform linux/arm64/v8 -t waziup/irrigation-prediction:dev --no-cache --pull --build-arg CACHEBUST=$(date +%s) --load .

### Push to dockerhub:
Issue the following cmd to push to dockerhub (you have to be logged in):
    
    docker push waziup/irrigation-prediction:dev

### Copy image to local Raspberry Pi:
Create a folder with the name of the app at the following path: "/var/lib/wazigate/apps/waziup.irrigation-prediction" and copy the files (docker-compose.yml & package.json) from the repository.

Issue the following cmd to push to dockerhub (you have to be logged in):

    docker save {id of image} | gzip | pv | ssh pi@{ip of rpi} docker load

