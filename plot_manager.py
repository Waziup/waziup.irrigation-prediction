import json
from plot import Plot
import os
import re
#import main # check whether 

# Plot related vars
Plots = {}                                          # This stores all plots
CurrentPlotId = 1                                   # CurrentPlotId to retrieve from Dict init with first plot   
CurrentPlotTab = 1                                  # init with first tab 
ConfigPath = 'config/current_config_plot1.json'     # init with first plot
Config_folder_path = "config/"                      # Folder with configfiles

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
    global CurrentPlotId, CurrentPlotTab, ConfigPath, Plots

    #files = readFiles()
    max_index = max(Plots.keys())

    # Point to config file of current plot
    try:
        # Should work if config is already set 
        newPlot = Plots[plot_nr_in_ui]
        ConfigPath = Config_folder_path + "current_config_plot" + str(newPlot.id) + ".json" #files[plot_nr_in_ui - 1] # -1 because the dict start with 1, not with 0
    # Newplot has been added as "last plot"
    except Exception as e:
        # Should only be the case if config was not set
        max_index = max(Plots.keys())
        newPlot = Plots[max_index]
        
        ConfigPath = newPlot.configPath

    # Set also changes in manager_class, TODO: redundant
    CurrentPlotId = newPlot.id
    CurrentPlotTab = plot_nr_in_ui

    return ConfigPath


# When App starts it looks through fromer plot configuration and reloads them, also creates object of a class that represents plots
def loadPlots():
    global Plots

    files = readFiles()
    amount = len(files)

    if amount == 0:
        plot_obj = Plot(1, ConfigPath)
        Plots[1] = plot_obj
        plot_obj.printPlotNumber()
    else:
        # Create class to manage tabs/plots
        for index, file in enumerate(files, start=1):  # Using enumerate to get index correctly
            plot_obj = Plot(index, os.path.join(Config_folder_path, file))
            Plots[index] = plot_obj
            plot_obj.printPlotNumber()  # Print for debugging

    return len(files)

# Add a plot during runtine TODO: finish
def addPlot(tabid):
    global Plots

    #files = readFiles()
    
    # retrieve the last plot and increment filename
    # try:
    #     newfilename = files[-1]
    # except Exception as e: # in case setup has never been run
    #     newfilename = 'config/current_config_plot0.json' # LOL
            
    next_number = Plots[max(Plots.keys())].id + 1
    newfilepath = os.path.join(Config_folder_path, "current_config_plot" + str(next_number) + ".json")
    # next_number = 0
    # match = re.search(r'plot(\d+)\.json$', newfilename)
    # if match:
    #     number = int(match.group(1))
    #     next_number = number + 1  # Increment number
    #     newfilename = re.sub(r'plot(\d+)(\.json)$', f'plot{next_number}.json', newfilename)   # Replace with new number

    # Create new plot
    plot_obj = Plot(int(tabid), newfilepath)
    Plots[len(Plots)+1] = plot_obj

    plot_obj.printPlotNumber()  # Print for debugging
    plot_obj.setState(True) # TODO: obsolete?

    return next_number, newfilepath

# Remove a plot from the list
def removePlot(plot_nr_to_be_removed):
    global Plots

    # Get current plot and remove
    plot_to_remove = getCurrentPlot()

    # Compare plot scope
    if plot_nr_to_be_removed is not CurrentPlotTab:
        print("Will remove plot number: ", plot_nr_to_be_removed)
    else:
        print("IndexError: Number of plot in frontend is different than backend, might have just deleted the wrong plot.")

    # Remove json config file TODO: DEBUG
    #os.remove(plot_to_remove.configPath)

    # Finally remove the plot from the list TODO: not array not suitable
    del Plots[CurrentPlotTab]

    return plot_to_remove.CurrentPlotTab, plot_to_remove.configPath

# Just access 
def getPlots():
    return Plots  # Returns the list of Plot objects

def getCurrentConfig():
    return Plots[CurrentPlotTab].configPath

def getCurrentPlot():
    return Plots[CurrentPlotTab]

def getCurrentPlotWithId(passed_id):
    for plot in Plots:
        if plot.id is passed_id:
            return plot
    return False

def setCurrentConfig(path):
    global Plots
    Plots[CurrentPlotTab].configPath = path
