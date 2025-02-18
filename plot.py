# Hold properties of a field in one class object
class Plot:
  prediction_thread = [] # TODO: only one
  training_thread = []
  currently_active = False

  def __init__(self, plot_number, configPath):
    self.plot_number = plot_number
    self.configPath = configPath

  # Just print some class properies
  def printPlotNumber(self):
    print("Current object is plot number: " + str(self.plot_number), ", with the path: " + self.configPath)

  # Set the current thread that runs prediction
  def setPredictionhread(thread):
    global prediction_thread
    prediction_thread.append(thread)

  # Set the current thread that runs training
  def setTrainingThread(thread):
    global training_thread
    training_thread.append(thread)

  # Redundant set active state
  def setState(state):
    global currently_active
    currently_active = state

  # Redundant get active state
  def getState():
    return currently_active