import sys
import datetime

import create_model

# Globals
# Timespan of hours 
TimeSpan = 12
OverThresholdAllowed = 1.2

# Mighty main fuction ;)
def main(currentSoilTension, threshold_timestamp) -> int:
    # Get threshold from config
    threshold = create_model.Current_config['Threshold']
    # time_spamk is set to 12h
    now = datetime.now().replace(microsecond=0)

    # "Weak" irrigation strategy
    # If threshold was met
    if currentSoilTension > threshold:
        print(f"Threshold: {threshold} of was reached with a value of {currentSoilTension}.")
        # If threshold + 20% -> irrigate
        if currentSoilTension * OverThresholdAllowed > threshold:
            print(f"Threshold: {threshold} was exceeded by 20%, irrigate immediatly!")
        # Threshold was met but predictions will not meet threshold in forecast horizon
        elif threshold_timestamp:
            print(f"Threshold: {threshold} was met but predictions will not meet threshold in forecast horizon")
            timestamp_end = datetime.strptime(threshold_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ") - datetime.timedelta(hours=TimeSpan)
            # have to calc next occurance of 
            if(datetime.datetime.strptime(threshold_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ") - datetime.timedelta(hours=TimeSpan)):
                print(f"")
    # Threshold was not met, so do not irrigate
    else:
        print(f"")






    


if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit