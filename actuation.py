import sys
from datetime import datetime, timedelta

import pandas as pd

import create_model

# Globals
# Timespan of hours 
TimeSpanOverThreshold = 12
OverThresholdAllowed = 1.2

# Find global max and min
def get_max_min(df, target_col='prediction_label'):
    # reset "new" index
    df = df.reset_index(inplace=False)
    
    # index
    global_min_index = df[target_col].idxmin()
    global_max_index = df[target_col].idxmax()
    # value
    global_min = df[target_col].min()
    global_max = df[target_col].max()
    
    print(global_min_index, "value:", global_min, global_max_index, "value:", global_max, "length:", global_max_index-global_min_index)
    print(global_min_index, "value:", df[target_col][global_min_index], global_max_index, "value:", df[target_col][global_max_index], "length:", global_max_index-global_min_index)

    return global_min_index, global_max_index, global_min, global_max


# Function to find next lower and higher value occurrence
def find_next_occurrences(df, column, threshold):
    # Start @current time
    # timezone = create_model.get_timezone(Current_config["Gps_info"]["lattitude"], Current_config["Gps_info"]["longitude"])
    idx = pd.Timestamp(datetime.now().replace(microsecond=0)).tz_localize('Europe/Berlin') #TODO: timezone is missing here, replace with timezone, uncomment above

    # Filter the DataFrame to include only rows with indices greater than or equal to 'idx'
    filtered_df = df[df.index >= idx]
    filtered_df = df[df.index <= idx + timedelta(hours=12)]

    # Further filter the DataFrame to include only rows where the specified column's value is less than the 'threshold'
    filtered_lower = filtered_df[filtered_df[column] < threshold]

    # Convert the filtered DataFrame's index to a list
    next_lower_idx = filtered_lower.index.tolist()
    if next_lower_idx:
        next_lower_idx = next_lower_idx[0]
    else:
        next_lower_idx = None

    # Find the next occurrence of a value higher than the threshold after the next lower index
    if next_lower_idx is not None:
        # Filter the DataFrame to include only rows with indices greater than or equal to 'next_lower_idx'
        filtered_df_higher = df[df.index >= next_lower_idx]

        # Further filter the DataFrame to include only rows where the specified column's value is greater than the 'threshold'
        filtered_higher = filtered_df_higher[filtered_df_higher[column] > threshold]

        # Convert the filtered DataFrame's index to a list
        next_higher_idx = filtered_higher.index.tolist()

        if next_higher_idx:
            next_higher_idx = next_higher_idx[0]
        else:
            next_higher_idx = None
    else:
        next_higher_idx = None

    return next_lower_idx.tz_convert('UTC').tz_localize(None) if next_lower_idx is not None else None, next_higher_idx.tz_convert('UTC').tz_localize(None) if next_higher_idx is not None else None # for that I will go to timezone hell


def irrigate():
    return 0


# Mighty main fuction TODO:capsulate
def main(currentSoilTension, threshold_timestamp, predictions) -> int:
    #####################################################################
    ## TODO: remove DEBUG vars                                          #
    # Get threshold from config                                         #
    threshold = 15#create_model.Current_config['Threshold']             #
    # set timestamp for debug reasons                                   #
    future_time =  datetime.now() + timedelta(hours=5)                  #
    threshold_timestamp = future_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ") #
    ## TODO: remove DEBUG vars                                          #
    #####################################################################
    # time_span is set to 12h
    now = datetime.now().replace(microsecond=0)
    then = now + timedelta(hours=TimeSpanOverThreshold)

    # "Weak" irrigation strategy
    # If threshold was met
    if currentSoilTension > threshold:
        print(f"Threshold: {threshold} was reached with a value of {currentSoilTension}.")
        # If threshold + 20% -> irrigate
        if currentSoilTension > threshold * OverThresholdAllowed:
            print(f"Threshold: {threshold} was exceeded by 20%, irrigate immediatly!")
            # Trigger irrigation
            e = irrigate()
            return e
        # Threshold was met but predictions will not meet threshold in forecast horizon
        elif not threshold_timestamp:
            print(f"Threshold: {threshold} was met but predictions will not meet threshold in forecast horizon")
            next_lower_idx, next_higher_idx = find_next_occurrences(predictions, 'prediction_label', threshold)
            if next_higher_idx:
                print(f"Threshold: {threshold} will be meet again in the in the next {TimeSpanOverThreshold} hours, irrigate now!")
                # Trigger irrigation
                e = irrigate()
                return e
            return 0
        elif threshold_timestamp:
            #and next occurance within 12h
            next_lower_idx, next_higher_idx = find_next_occurrences(predictions, 'prediction_label', threshold)
            if next_higher_idx:
                print(f"Threshold: {threshold} will be meet again in the next {TimeSpanOverThreshold} hours, irrigate now!")
                # Trigger irrigation
                e = irrigate()
                return e
            else:
                print(f"Threshold was not met yet.")
                return 0
    # Threshold was not met, so do not irrigate
    else:
        print(f"Threshold: {threshold} is not reached, do not irrigate.")
        return 0






    


if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit