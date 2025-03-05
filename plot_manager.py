from plot import Plot
import os
import re
import main # check whether 

# Plot related vars
Plots = {}                                      # This stores all plots
CurrentPlot = 1                                 # init with first plot
ConfigPath = 'config/current_config_plot1.json' # init with first plot
Config_folder_path = "config/"                  # Folder with configfiles

# Array of active threads TODO: if training started kill other threads.(formerly done in main.py)
#Threads = []
#ThreadId = 0
#Restart_time = 1800 # DEBUG 1800 ~ 30 min in s

# Just read directory and retrieve filenames in sorted manner
def readFiles():
    try:
        files = [f for f in os.listdir(Config_folder_path) if os.path.isfile(os.path.join(Config_folder_path, f))]
        files.sort()
    except Exception as e:
        files = []

    return files

# Set the current ṕlot
def setPlot(plot_nr_in_ui):
    global CurrentPlot, ConfigPath, Plots

    files = readFiles()

    # Point to config file of current plot
    try:
        Plots[plot_nr_in_ui].configPath = Config_folder_path + files[plot_nr_in_ui - 1]
    # There is no file, iterate the highest plot number present in /config folder
    except Exception as e:
        match = re.search(r'plot(\d+)\.json$', Plots[len(Plots)].configPath)
        if match:
            number = int(match.group(1))
            Plots[plot_nr_in_ui].configPath  = re.sub(r'plot(\d+)(\.json)$', f'plot{number}.json', Plots[len(Plots)].configPath) 

    # Set also changes in manager_class, TODO: redundant
    CurrentPlot = plot_nr_in_ui
    ConfigPath = Plots[plot_nr_in_ui].configPath

    # Get config from json and load data to vars TODO: check duplicate call of getConfigFromFile, also called via API by frontend!!!
    return main.getConfigFromFile()


# When App starts it looks through fromer plot configuration and reloads them, also creates object of a class that represents plots
def loadPlots():
    global Plots

    files = readFiles()

    # Create class to manage tabs/plots
    for index, file in enumerate(files, start=1):  # Using enumerate to get index correctly
        plot_obj = Plot(index, os.path.join(Config_folder_path, file))
        Plots[index] = plot_obj
        plot_obj.printPlotNumber()  # Print for debugging

    return len(files)

# Add a plot during runtine TODO: finish
def addPlot():
    global Plots

    files = readFiles()
    
    # retrieve the last plot and increment filename
    try:
        newfilename = files[-1]
    except Exception as e: # in case setup has never been run
        newfilename = 'config/current_config_plot0.json' # LOL

    next_number = 0
    match = re.search(r'plot(\d+)\.json$', newfilename)
    if match:
        number = int(match.group(1))
        next_number = number + 1  # Increment number
        newfilename = re.sub(r'plot(\d+)(\.json)$', f'plot{next_number}.json', newfilename)   # Replace with new number

    # Create new plot
    plot_obj = Plot(next_number, Config_folder_path + newfilename)
    Plots[next_number] = plot_obj

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
    return Plots[CurrentPlot].configPath

def getCurrentPlot():
    return Plots[CurrentPlot]

def setCurrentConfig(path):
    Plots[CurrentPlot].configPath = path
