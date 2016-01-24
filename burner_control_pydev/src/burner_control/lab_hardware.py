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
      location (double): location of the instrument from the combustor
    """
    
    self.location = location
  
  def get_location(self):
    """
    Gets the location property of the instrument.
    
    Returns:
      double: location of the instrument from the combustor
    """

    return self.location
  
class Combustor():
  '''
  Defines the Combustor class, which is basically the containing class for a lot
  of the experimental setup. 
  '''

  def __init__(self, p_atm, t_step, t_step_ctrl):
    '''
    Constructor. Adds components of the combustor to its component list, sets
    the atmospheric pressure, and gets everything ready to go (theoretically).
    
    Args:
      p_atm (double): atmospheric pressure in psi
      t_step (double): time step for system simulation
    '''
    
    # Initialize constants
    self.p_atm = p_atm
    self.t_step = t_step
    self.time = 0.0  # initialize current time to zero
    K = 0.1
    offset = 1.0
    K_mfcs = [10.0, 5.0, 2.0, 1.0]  # MFC gains
    tau_mfcs = [1.5, 2.5, 3.0, 1.4]  # MFC time constants
    td_mfcs = [0.2, 0.1, 0.4, 0.3]  # MFC time delays
    y0 = [0.0, 0.0, 0.0]  # initial value
    
    # Initialize sensor list
    self.sensor_list = [StaticSensor(model=lambda y: sim_functions.static_model(y, K, offset),
                                     location=0.0),
                        DynamicSensor(model=tf1, location=1.0)]
    #TODO initialize others
    #TODO figure out transfer functions!
    
    # Initilaize flame
    self.flame = Flame(operating_map=lambda p: sim_functions.one_sphere(p, radius=2.0))
    #TODO figure out operating space maps!
    
    # Initialize mass flow controller (MFC) list
    mfc_list = [MFC(lambda t, y, u: sim_functions.first_order_delay(t, y, u, tau, td),
                    lambda y: sim_functions.first_order_output(y, K, td), y0)
                for K, tau, td in zip(K_mfcs, tau_mfcs, td_mfcs)]
    # Initialize control law list
    #TODO figure this out!
    
    # Initialize the controller (which will contain the MFC list)
    self.controller = Controller(mfc_list, control_law_list, t_step_ctrl)
    
  def update(self):
    """
    Update the experimental apparatus by calling the update functions for each
    component sequentially.
    """
    
    self.controller.update(mass_flow_des, self.t_step, self.time)
    #TODO figure out how to update desired mass flow rate
    
    flame_snapshot = self.flame.update(self.controller.get_output())
    #TODO figure out what the flame has to return
    
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
    self.state = False
    
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
      y0 (double array) initial state of the ODE.
    """
    
    self.ode = integrate.ode(ode_fcn)
    self.ode.set_integrator("dopri5")  # RK45 solver
    self.ode.set_initial_value(y0)  # initial state, initial t = 0.0
    self.output_fcn = output_fcn
    
  def get_time(self):
    """
    Get the internal time used by the MFC ODE.
    
    Returns:
      double: simulation time of the ODE
    """
    
    return self.ode.t
    
  def get_output(self):
    """
    Get the current output of the MFC through its output function.
    
    Returns:
      double: ODE output from the ODE output function
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
      input (double): controller input to the MFC ODE (volts)
      t_step (double): time step used for simulation
    
    Returns:
      double: mass flow rate (LPM or something)
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
      location (double): location of the sensor from the combustor
    """
    
    self.model = model
    self.reading = 0.0
    #TODO figure out how to make use of the location
    super(StaticSensor, self).__init__(location)  # parent keeps location
    
  def get_output(self):
    """
    Get the output of the sensor.
    
    Returns:
      double: output of the sensor
    """
    
    return self.reading
  
  def update(self, in_value):
    """
    Update the output of the static sensor.
    
    Args:
      in_value (double): state that the sensor is reading
    
    Returns:
      double: output of the sensor
    """
    
    self.reading = self.model(in_value)
    return self.reading

class DynamicSensor(Instrument):
  """Defines the dynamic sensor object."""
  
  def __init__(self, model, location):
    """
    Constructor
    
    Args:
      model (undecided): some model defining the response of the sensor to
        stimulus
      location (double): location of the sensor from the combustor
    """
    
    #TODO maybe use an ODE to define the sensor model, like in MFC class?
    self.model = model
    self.reading = 0.0
    #TODO figure out how to make use of the location
    super(DynamicSensor, self).__init__(location)  # parent keeps location
  
  def get_output(self):
    """
    Get the instant output state of the sensor.
    
    Returns:
      double: single-value instant reading of the sensor
    """
    
    return self.reading
  
  def get_time_series(self):
    """Output the response of the sensor as a time series."""
    
    pass
  
  def get_power_spectrum(self):
    """Output the response of the sensor as a power spectrum."""
    
    pass
  
  def update(self, in_value, t_step):
    """
    Update the response of the sensor.
    
    Args:
      in_value (double): state that the sensor is reading
      t_step (double): time step for simulation
    
    Returns:
      double: single-value instant reading of the sensor
    """
    
#     self.reading = self.model(in_value, t_step)
#     return self.reading
    pass
  
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
      t_step_ctrl (double): iteration rate for the control law. Should be
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
      list, double: measurements from each controlled object
    """
    
    if isinstance(self.mfc_list, MFC):
      return [self.mfc_list.get_output()]  # single MFC case
    else:
      return [mfc.get_output() for mfc in self.mfc_list]  # multiple MFC case
  
  def get_time(self):
    """
    Get the simulation time of the first MFC in mfc_list.
    
    Returns:
      double: simulation time of the first MFC in mfc_list
    """
    
    if isinstance(self.mfc_list, MFC):
      return self.mfc_list.get_time()  # single MFC case
    else:
      return self.mfc_list[0].get_time()  # multiple MFC case
    
  def update(self, mass_flow_des, t_step):
    """
    Update the controller and its MFCs.
    
    Args:
      mass_flow_des (double array): desired mass flow rates for MFCs (LPM)
      t_step (double): time step used for simulation (seconds)
      time (double): current simulation time (seconds)
    
    Returns:
      bool: True if successful, False otherwise
    """
    
    # Update the control effort based on the current and desired state when the
    # simulation time reaches another controller period.
    if self.get_time() % self.t_step_ctrl < 1.0:
      #TODO test for length of mass_flow_desired
      self.u_ctrl = [ctrl_law(mfc.get_output(), ref) for ctrl_law, mfc, ref
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
    
    Kwargs:
      Q (double, default=1e-5): estimate of process variance
      R (double, default=1e-3): estimate of measurement variance
    """
    
    self.xhat_minus = np.zeros(A.shape[1])
    self.xhat = self.xhat_minus
    self.P_minus = P
    self.P = P
    self.Q = Q
    self.R = R
    self.A = A
    self.B = B
    self.C = C
  
  def update(self, y, u):
    """
    Updates the state of the Kalman filter.
    
    Args:
      z (double) update measurement
    
    Returns:
      array, double: [xhat, P], current state and error estimate
    """
    
    #TODO implement an actual KF from papers
    # Store previous values
    self.xhat_minus = self.xhat
    self.P_minus = self.P
    
    # Measurement update
    x_apri = self.A.dot(self.xhat_minus) + self.B.dot(u)
    P_apri = self.A.dot(self.P_minus).dot(self.A.T) + self.Q
    inv = np.linalg.inv(self.C.dot(P_apri).dot(self.C.T) + self.R)
    K = P_apri.dot(self.C.T).dot(inv)
    self.xhat = x_apri + K.dot(y - self.C.dot(x_apri))
    self.P = (np.identity(self.P.shape) - K.dot(self.C)).dot(P_apri)
    return [self.xhat, self.P]
    