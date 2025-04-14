# WaziUps WaziApp: irrigation-prediction
This application (WaziApp) for the WaziGate predicts irrigation times with help of soil sensors and a weather API (open-meteo). A non-technical user guide with general instructions can be obtained from [here](help/user_guide.md).

## How to install

Visit the WaziGates UI and open the App section. Press the plus button and click install custom app. Type into the textbox "waziup/irrigation-prediction:latest" to download/install the application from dockerhub. 

## How to change run configuration

For debugging there are 3 run configurations possible:

- **Production mode:** run as WaziApp in a docker container on the WaziGate
- **Local debugging environment:** Debug in Visual Studio with wazigate-edge and wazigate-dashboard running locally 
- **Local debugging environment against local gateway:** Debug in Visual Studio with wazigate-edge and wazigate-dashboard running locally, using the API of a WaziGate that has its API exposed (e.g. local network).

You can change the "run configuration" in the [".env" file](.env), just choose one of the three available options.

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

