from plot import Plot
import os
import re

# Plot related vars
Plots = []  # This stores all plots
CurrentPlot = 1
ConfigPath = 'config/current_config_plot1.json' #init with first plot
Config_folder_path = "config/"

# Array of active threads TODO: if training started kill other threads.
#Threads = []
#ThreadId = 0
#Restart_time = 1800 # DEBUG 1800 ~ 30 min in s


# When App starts it looks through fromer plot configuration and reloads them, also creates object of a class that represents plots
def loadPlots():
    global Plots

    files = [f for f in os.listdir(Config_folder_path) if os.path.isfile(os.path.join(Config_folder_path, f))]

    # sort them by index
    files.sort()
    #print(files)

    # Create class to manage tabs/plots
    for index, file in enumerate(files, start=1):  # Using enumerate to get index correctly
        plot_obj = Plot(index, Config_folder_path + file)
        Plots.append(plot_obj)
        plot_obj.printPlotNumber()  # Print for debugging

    return len(files)

# Add a plot during runtine TODO: finish
def addPlot():
    global Plots

    files = [f for f in os.listdir(Config_folder_path) if os.path.isfile(os.path.join(Config_folder_path, f))]

    # sort them by index
    files.sort()
    
    # retrieve the last plot and increment filename
    newfilename = files[-1]
    next_number = 0
    match = re.search(r'plot(\d+)\.json$', newfilename)
    if match:
        number = int(match.group(1))
        next_number = number + 1  # Increment number
        newfilename = re.sub(r'plot(\d+)(\.json)$', f'plot{next_number}.json', newfilename)   # Replace with new number


    plot_obj = Plot(next_number, Config_folder_path + newfilename)
    Plots.append(plot_obj)
    plot_obj.printPlotNumber()  # Print for debugging
    plot_obj.setState(True)

    return next_number, newfilename

# Remove a plot from the list
def removePlot(plot_nr_to_be_removed):
    global Plots

    # Get current plot and remove
    plot_to_remove = getCurrentPlot()

    # Compare plot scope
    if plot_nr_to_be_removed is not CurrentPlot:
        print("Will remove plot number: ", plot_nr_to_be_removed)
    else:
        print("IndexError: Number of plot in frontend is different than backend, might have just deleted the wrong plot.")

    # Remove json config file
    os.remove(plot_to_remove.configPath)

    # Finally remove the plot from the list TODO: not array not suitable
    del Plots[CurrentPlot-1]

    return plot_to_remove.CurrentPlot, plot_to_remove.configPath

# Just access 
def getPlots():
    return Plots  # Returns the list of Plot objects

def getCurrentConfig():
    return Plots[CurrentPlot-1].configPath

def getCurrentPlot():
    return Plots[CurrentPlot-1]

def setCurrentConfig(path):
    Plots[CurrentPlot-1].configPath = path

def setCurrentPlot(nr):
    global CurrentPlot

    CurrentPlot = nr
