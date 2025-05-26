from plot import Plot
import os
import threading

# Global variables
Plots = {}                                          # This stores all plots in a shared dictionary      
CurrentPlotId = 1                                   # CurrentPlotId to retrieve from Dict init with first plot   
CurrentPlotTab = 1                                  # init with first tab 
ConfigPath = 'config/current_config_plot1.json'     # init with first plot
Config_folder_path = "config/"                      # Folder with configfiles

# Array of active threads TODO: if training started kill other threads.(formerly done in main.py)
Threads = []
ThreadId = 0

# Just read directory and retrieve filenames in sorted manner
def readFiles():
    try:
        files = [f for f in os.listdir(Config_folder_path) if os.path.isfile(os.path.join(Config_folder_path, f))]
        files.sort()
    except Exception as e:
        files = []

    return files

# Set the current ṕlot
def setPlot(plot_nr_tab):
    global CurrentPlotId, CurrentPlotTab, ConfigPath

    # Get the plot object from the dictionary
    currentPlot = Plots[plot_nr_tab] 

    # Point to config file of current plot
    try:
        # Should work if config is already set 
        ConfigPath = Config_folder_path + "current_config_plot" + str(currentPlot.id) + ".json" #files[plot_nr_in_ui - 1] # -1 because the dict start with 1, not with 0
    # Newplot has been added as "last plot"
    except Exception as e:
        # Should never be the case, but just in case
        max_index = max(Plots.keys())
        newPlot = Plots[max_index]
        ConfigPath = newPlot.configPath
        print (f"Error setting plot: {e}. Using last plot's config path: {ConfigPath}")

    # Set also changes in manager_class, TODO: redundant
    CurrentPlotId = currentPlot.id
    CurrentPlotTab = getCurrentPlotNumberWithId(currentPlot)

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
def addPlot(tabNumber):
    global Plots

    #files = readFiles()
    
    # retrieve the last plot and increment filename
    # try:
    #     newfilename = files[-1]
    # except Exception as e: # in case setup has never been run
    #     newfilename = 'config/current_config_plot0.json' # LOL
            
    next_number = max(plot.id for plot in Plots.values()) + 1
    newfilepath = os.path.join(Config_folder_path, "current_config_plot" + str(next_number) + ".json")
    # next_number = 0
    # match = re.search(r'plot(\d+)\.json$', newfilename)
    # if match:
    #     number = int(match.group(1))
    #     next_number = number + 1  # Increment number
    #     newfilename = re.sub(r'plot(\d+)(\.json)$', f'plot{next_number}.json', newfilename)   # Replace with new number

    # Create new plot
    plot_obj = Plot(int(tabNumber), newfilepath)
    Plots[len(Plots)+1] = plot_obj

    plot_obj.printPlotNumber()  # Print for debugging
    plot_obj.setState(True) # TODO: obsolete?

    return tabNumber, newfilepath

# Remove a plot from the list
def removePlot(plot_nr_to_be_removed):
    global Plots, CurrentPlotTab, CurrentPlotId

    # Get current plot and remove -> formerly was getCurrentPlot()
    plot_to_remove = Plots[CurrentPlotTab]
    #plot_to_remove = getCurrentPlotWithId(plot_nr_to_be_removed)

    # Compare plot scope
    if plot_nr_to_be_removed is CurrentPlotTab:
        print("Will remove plot number: ", plot_nr_to_be_removed)
    else:
        print("IndexError: Number of plot in frontend is different than backend, might have just deleted the wrong plot.")

    try:
        # Remove json config file TODO: DEBUG
        os.remove(plot_to_remove.configPath)
    except Exception as e:
        print(f"Error removing file, most likely it is not being created. Error: {e}")

    # Finally remove the plot from the list TODO: not array not suitable
    if removePlotWithId(plot_nr_to_be_removed):
        print(f"Plot {plot_nr_to_be_removed} removed successfully.")
    else: 
        print(f"Failed to remove plot {plot_nr_to_be_removed}.")

    # Set plot to former plot
    CurrentPlotTab = CurrentPlotTab - 1
    if CurrentPlotTab < 1:
        CurrentPlotTab = 1

    # Assign former plot to be current plot (id)
    try:
        CurrentPlotId = Plots[CurrentPlotTab].id
    except Exception as e:
        print(f"Error setting current plot ID: {e}")
        CurrentPlotId = False

    return plot_nr_to_be_removed, plot_to_remove.configPath

# Just access 
def getPlots():
    return Plots  # Returns the list of Plot objects

def getCurrentConfig():
    return Plots[CurrentPlotTab].configPath

def getCurrentPlot():
    return Plots[CurrentPlotTab]

# def getCurrentPlot():
#     print(f"Into get cueernt plot with id: {CurrentPlotTab}")
#     print(f"[{multiprocessing.current_process().name}] Trying to acquire plot_lock...")
#     with plot_lock:
#         print(f"[{multiprocessing.current_process().name}] Lock acquired.")
#         plot = Plots[CurrentPlotTab]
#     print(f"[{multiprocessing.current_process().name}] Lock released.")
#     return plot

def getCurrentPlotNumberWithId(currentPlot):
    global Plots
    i = 1
    for plot in Plots.values():
        if plot.id == currentPlot.id:
            return i
        i += 1
    return 1

def getCurrentPlotWithId(passed_id):
    for plot in Plots.values():
        if plot.id == passed_id:
            return plot
    return False

def removePlotWithId(passed_id):
    global Plots
    i = 1
    for plot in Plots.values():
        if plot.id == passed_id:
            del Plots[i]
            return True
        i += 1
    return False

def setCurrentConfig(path):
    global Plots
    Plots[CurrentPlotTab].configPath = path
