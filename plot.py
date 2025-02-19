# Hold properties of a field in one class object
class Plot:
  # Threads that perfom the actual training
  Training_thread = None
  Prediction_thread = None

  # Flags that toggle  
  TrainingFinished = False
  CurrentlyTraining = False
  
  # Redundant, as saved in manager
  Currently_active = False

  # Class init, called when created in UI
  def __init__(self, plot_number, configPath):
    self.plot_number = plot_number
    self.configPath = configPath

    # Variables that were global before, now plot-specific TODO: implement everywhere
    self.device_and_sensor_ids_moisture = []
    self.device_and_sensor_ids_temp = []
    self.device_and_sensor_ids_flow = []
    self.gps_info = ""
    self.sensor_kind = "tension"
    self.sensor_unit = ""
    self.slope = 0
    self.threshold = 0
    self.irrigation_amount = 0
    self.look_ahead_time = 0
    self.start_date = ""
    self.period = 0
    self.train_period_days = 1
    self.predict_period_hours = 6
    self.soil_type = ""
    self.permanent_wilting_point = 40
    self.field_capacity_upper = 30
    self.field_capacity_lower = 10
    self.saturation = 0
    self.training_thread = None
    self.prediction_thread = None
    self.training_finished = False
    self.currently_training = False
    self.currently_active = False

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
    global Currently_active
    Currently_active = state

  # Redundant get active state
  def getState():
    return Currently_active
  
  # Set training thread
  def setTrainingThread(thread):
    global Training_thread
    Training_thread = thread

  # Get training thread
  def getTrainingThread(thread):
    return Training_thread
  
  # Set prediction thread
  def setPredictionThread(thread):
    global Prediction_thread
    Prediction_thread = thread

  # Get prediction thread
  def getPredictionThread(thread):
    return Prediction_thread
