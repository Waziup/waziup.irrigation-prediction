from plot import Plot
import os
import re

Plots = []  # This stores all plots
CurrentPlot = 1


# When App starts it looks through fromer plot configuration and reloads them, also creates object of a class that represents plots
def loadPlots():
    global Plots

    folder_path = "config/"
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # sort them by index
    files.sort()
    #print(files)

    # Create class to manage tabs/plots
    for index, file in enumerate(files, start=1):  # Using enumerate to get index correctly
        plot_obj = Plot(index, folder_path + file)
        Plots.append(plot_obj)
        plot_obj.printPlotNumber()  # Print for debugging

    return len(files)

# Add a plot during runtine TODO: finish
def addPlot():
    global Plots

    folder_path = "config/"
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

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


    plot_obj = Plot(next_number, folder_path + newfilename)
    Plots.append(plot_obj)
    plot_obj.printPlotNumber()  # Print for debugging
    plot_obj.setState(True)

    return next_number, newfilename

def create_new_plot():
    global Plots
    folder_path = "config/"

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()

    # Populate the list
    for index, file in enumerate(files, start=1):
        plot_obj = Plot(index, file)
        Plots.append(plot_obj)

def getPlots():
    return Plots  # Returns the list of Plot objects