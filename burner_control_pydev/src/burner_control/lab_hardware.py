'''
Created on Jan 7, 2016

@author: nathantoner
'''

from scipy import integrate
import numpy as np
from burner_control import sim_functions


class Instrument():
  """
  Defines the instrument class. All of the different measurement apparatus that
  we simulate will be children of this class.
  """
  def __init__(self, location):
    """
    Constructor.
    
    Args:
      location (float): location of the instrument from the combustor
    """
    
    self.location = location
  
  def get_location(self):
    """
    Gets the location property of the instrument.
    
    Returns:
      float: location of the instrument from the combustor
    """

    return self.location
  
class Combustor():
  '''
  Defines the Combustor class, which is basically the containing class for a lot
  of the experimental setup. 
  '''

  def __init__(self, p_atm, t_step, t_step_ctrl, sensor_list, mfc_list,
               control_law_list):
    '''
    Constructor. Adds components of the combustor to its component list, sets
    the atmospheric pressure, and gets everything ready to go (theoretically).
    
    Args:
      p_atm (float): atmospheric pressure in psi
      t_step (float): time step for system simulation
      sensor_list (list, Instrument): list of sensor objects
      mfc_list (list, MFC): list of MFC objects
      control_law_list (list, function): list of control law functions
    '''
    
    # Initialize constants
    self.p_atm = p_atm
    self.t_step = t_step
    self.time = 0.0  # initialize current time to zero
    self.flame = Flame(operating_map=lambda p: sim_functions.one_sphere(p, radius=2.0))
    self.controller = Controller(mfc_list, control_law_list, t_step_ctrl)
    
  def update(self, mass_flow_des):
    """
    Update the experimental apparatus by calling the update functions for each
    component sequentially.
    
    Args:
      mass_flow_des (ndarray): desired mass flow rates for all MFCs
    """
    
    self.controller.update(mass_flow_des, self.t_step, self.time)
    #TODO figure out how to update desired mass flow rate
    
    flame_snapshot = self.flame.update(self.controller.get_output())
    #TODO figure out what the flame has to return. Or we could just pass the
    # flame to sensors.
    
    for sensor in self.sensor_list:
      sensor.update(flame_snapshot, self.t_step)
    
    self.time += self.t_step

class Flame():
  """Defines the flame held within the combustor."""
  
  def __init__(self, operating_map, model=None):
    """
    Constructor
    
    Args:
      map (function): maps out the actual stable/unstable regions of the
        system's operating space
      model (undecided): model relating some flame output to the current
        operating point of the flame
    """
    
    self.operating_map = operating_map
    self.model = model  #TODO figure out what to do with the model
    self.state = False  # no flame to start
    
  def ignite(self):
    """Lights the flame by setting its state to True."""
    
    self.state = True
  
  def blowout(self):
    """Extinguishes the flame by setting its state to False."""
    
    self.state = False
    
  def get_state(self):
    """
    Gets the flame state.
    
    Returns:
      bool: True if flame, False if no flame
    """
    
    return self.state
    
  def update(self, operating_point):
    """
    Updates the state of flame and outputs appropriate stuff.
    
    Args:
      operating_point (list): current state of the MFCs coming from the
        Controller.get_output() method
    
    Returns:
      bool: True of flame, False if no flame
      flame_snapshot (undecided): I am not sure what this should be yet
    """
    
    #TODO update the flame model and return some snapshot of the physical
    # combustor space maybe?
    
    # Update the flame state based on operating map
    if not self.state and self.operating_map(operating_point):
      self.ignite()
    if self.state and not self.operating_map(operating_point):
      self.blowout()
    return self.state
      
class MFC():
  """Defines the mass flow controller object."""
  
  def __init__(self, ode_fcn, output_fcn, y0):
    """
    Constructor
    
    Args:
      ode_fcn (function) ODE that governs the dynamics of the system.
      output_fcn (function) function that converts state from ODE into
        meaningful system output, i.e. measurement function
      y0 (float array) initial state of the ODE.
    """
    
    self.ode = integrate.ode(ode_fcn)
    self.ode.set_integrator("dopri5")  # RK45 solver
    self.ode.set_initial_value(y0)  # initial state, initial t = 0.0
    self.output_fcn = output_fcn
    
  def get_time(self):
    """
    Get the internal time used by the MFC ODE.
    
    Returns:
      float: simulation time of the ODE
    """
    
    return self.ode.t
    
  def get_output(self):
    """
    Get the current output of the MFC through its output function.
    
    Returns:
      float: ODE output from the ODE output function
    """
    
    return self.output_fcn(self.ode.y)
  
  def get_state(self):
    """
    Get the current internal state of the MFC ODE.
    
    Returns:
      ndarray: internal state of the ODE model
    """
    
    return self.ode.y
    
  def update(self, input_val, t_step):
    """
    Updates the output of the MFC step-by-step.
    
    Args:
      input (float): controller input to the MFC ODE (volts)
      t_step (float): time step used for simulation
    
    Returns:
      float: mass flow rate (LPM or something)
    """
    
    self.ode.set_f_params(input_val)
    self.ode.integrate(self.ode.t + t_step)
    return self.ode.successful()

class StaticSensor(Instrument):
  """Defines a static sensor object."""
  
  def __init__(self, model, location):
    """
    Constructor
    
    Args:
      model (function): describes output of sensor as function of input
      location (float): location of the sensor from the combustor
    """
    
    self.model = model
    self.reading = 0.0
    #TODO figure out how to make use of the location
    super(StaticSensor, self).__init__(location)  # parent keeps location
    
  def get_output(self):
    """
    Get the output of the sensor.
    
    Returns:
      float: output of the sensor
    """
    
    return self.reading
  
  def update(self, in_value):
    """
    Update the output of the static sensor.
    
    Args:
      in_value (float): state that the sensor is reading
    
    Returns:
      float: output of the sensor
    """
    
    self.reading = self.model(in_value)
    return self.reading

class DynamicSensor(Instrument):
  """Defines the dynamic sensor object."""
  
  def __init__(self, model, y0, location, rate):
    """
    Constructor
    
    Args:
      model (function): ODE defining the response of the sensor to stimulus
      y0 (float array): initial condition of the system's model
      location (float): location of the sensor from the combustor, mm
      rate (float): sample rate of the sensor, Hz
    """
    
    #TODO figure out an ODE dynamic sensor model that works
    #TODO maybe use this class to just simulate some noise signal at sensor
    # rate with frequency-dependent amplitude based on location in the operating
    # space of the combustor?
    self.model = integrate.ode(model)
    self.ode.set_integrator("dopri5")  # RK45 solver
    self.ode.set_initial_value(y0)  # initial state, initial t = 0.0
    self.rate = rate
    self.reading = []
    self.time = []
    #TODO figure out how to make use of the location
    super(DynamicSensor, self).__init__(location)  # parent keeps location
  
  def get_time_series(self):
    """
    Output the response of the sensor as a time series.
    
    Returns:
      reading (float array): time series reading of the sensor for current sim
        time step
      time (float array): time value for each reading starting at 0.0, sec
    """
    
    return self.reading, self.time
  
  def get_power_spectrum(self):
    """
    Output the response of the sensor as a power spectrum.
    
    Returns:
      (complex array): power spectrum of time series, obtained by FFT
      (float array): frequency values for power spectrum
    """
    
    return np.fft.fft(self.reading), np.fft.fftfreq(self.time.shape[-1])
  
  def update(self, in_value, t_step):
    """
    Update the response of the sensor. Note that if the sensor's sample rate is
    greater than the input t_step value, you will still get a single sample
    every time update is called. In general, update will return a time series
    that is max(1, floor(t_step/self.rate)) elements long.
    
    Args:
      in_value (float): state that the sensor is reading
      t_step (float): time step for simulation
    
    Returns:
      reading (float array): time series reading of the sensor for current sim
        time step
      time (float array): time value for each reading, starting at 0.0
    """
    
    #TODO
    # Maybe make a time series here and store it as self.reading. Basically
    # simulate the response of the sensor at the sensor's sample rate for the
    # time t_step, store that time series, and then we can either return that
    # time series or return some analysis thereof.
    
    temp_time = 0.0
    self.reading = []
    
    while temp_time < t_step:
      self.time.append(temp_time)
      self.model.set_f_params(in_value)
      self.model.integrate(self.model.t + self.rate)
      if not self.model.successful():
        break
      self.reading.append(self.model.y)
      temp_time += self.rate
    return self.reading, self.time
  
class Controller():
  """
  Defines the controller class that will be used to control the MFCs to follow
  a prescribed input trajectory.
  """
  
  def __init__(self, mfc_list, control_law_list, t_step_ctrl):
    """
    Constructor.
    
    Args:
      mfc_list (list of MFC): MFC objects defining the mass flow controllers
      control_law_list (list of functions): control laws that apply to each MFC
        in the mfc_list
      t_step_ctrl (float): iteration rate for the control law. Should be
        longer than the simulation time step.
    """
    
    self.t_step_ctrl = t_step_ctrl  # time step for updating the control law
    self.mfc_list = mfc_list
    self.control_law_list = control_law_list
    if isinstance(mfc_list, MFC):
      self.u_ctrl = [0]  # single MFC case
    else:
      self.u_ctrl = [0]*len(mfc_list)  # multiple MFC case
    
  def get_output(self):
    """
    Returns some indication of the states of each object that the controller is
    controlling.
    
    Returns:
      list, float: measurements from each controlled object
    """
    
    if isinstance(self.mfc_list, MFC):
      return [self.mfc_list.get_output()]  # single MFC case
    else:
      return [mfc.get_output() for mfc in self.mfc_list]  # multiple MFC case
  
  def get_time(self):
    """
    Get the simulation time of the first MFC in mfc_list.
    
    Returns:
      float: simulation time of the first MFC in mfc_list
    """
    
    if isinstance(self.mfc_list, MFC):
      return self.mfc_list.get_time()  # single MFC case
    else:
      return self.mfc_list[0].get_time()  # multiple MFC case
    
  def update(self, mass_flow_des, t_step):
    """
    Update the controller and its MFCs.
    
    Args:
      mass_flow_des (float array): desired mass flow rates for MFCs (LPM)
      t_step (float): time step used for simulation (seconds)
      time (float): current simulation time (seconds)
    
    Returns:
      bool: True if successful, False otherwise
    """
    
    # Update the control effort based on the current and desired state when the
    # simulation time reaches another controller period.
    if self.get_time() % self.t_step_ctrl < 1.0:
      #TODO test for length of mass_flow_desired
      self.u_ctrl = [ctrl_law(ref - mfc.get_output()) for ctrl_law, mfc, ref
                     in zip(self.control_law_list, self.mfc_list, mass_flow_des)]
    
    # Update the MFC using the current control effort.
    for mfc, u in zip(self.mfc_list, self.u_ctrl):
      if not mfc.update(u, t_step):  # run the update and check that it completed successfully
        return False
    
    return True

class KalmanFilter():
  """
  Defines a Kalman filter class for reducing noise in measurements.
  """
  
  def __init__(self, A, B, C, Q, R, P):
    """
    Constructor.
    
    Args:
      A (ndarray): state update matrix, x_k+1 = A*x_k + B*u_k
      B (ndarray): input matrix, x_k+1 = A*x_k + B*u_k
      C (ndarray): measurement update matrix, y_k = C*x_k + D*u_k
      Q (ndarray): estimate of process variance
      R (ndarray or float): estimate of measurement variance
      P (ndarray): initial estimate of error covariance
    """
    
    self.xhat_minus = np.zeros((A.shape[1], 1))
    self.xhat = self.xhat_minus
    self.P_minus = P
    self.P = P
    self.Q = Q
    self.R = R
    self.A = A
    self.B = B
    self.C = C
  
  def get_output(self):
    """
    Get the current output (prediction) of the Kalman filter.
    
    Returns:
      ndarray: KF-predicted state of the observed system
    """
    
    return self.xhat
  
  def get_err_cov(self):
    """
    Get the estimated error covariance matrix of the Kalman filter.
    
    Returns:
      ndarray: estimated error covariance matrix, P
    """
    
    return self.P
  
  def update(self, y, u):
    """
    Updates the state of the Kalman filter.
    
    Args:
      z (ndarray or float): update measurement
    
    Returns:
      ndarray: xhat, current state estimate
    """
    
    #TODO maybe implement KF without storing _minus states or _apri states
    # Store previous values
    self.xhat_minus = self.xhat
    self.P_minus = self.P
    
    # Measurement update
    x_apri = self.A.dot(self.xhat_minus) + self.B.dot(u)
    P_apri = self.A.dot(self.P_minus).dot(self.A.T) + self.Q
    inv = self.C.dot(P_apri).dot(self.C.T) + self.R
    if inv.shape != ():
      inv = np.linalg.inv(inv)
    else:
      inv = 1/inv
    K = P_apri.dot(self.C.T).dot(inv)
    self.xhat = x_apri + K.dot(y - self.C.dot(x_apri))
    self.P = (np.identity(self.P.shape[0]) - K.dot(self.C)).dot(P_apri)
    return self.xhat
    