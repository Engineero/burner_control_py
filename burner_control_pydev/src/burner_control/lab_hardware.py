'''
Created on Jan 7, 2016

@author: nathantoner
'''

import numpy as np
from scipy import integrate

def one_sphere(point, radius=1.0):
    """
    Checks whether points are inside or outside of a hypersphere.
    
    Args:
        point (double) array representing test condition
        radius (double, default=1.0) radius of hypersphere to test
    
    Return:
        1 if point is outside of the sphere (good)
        0 if point is inside of or on the sphere (bad)
    """
    
    return 1 if sum([num**2 for num in point]) > radius**2 else 0
  
def first_order_delay(t, y, u, tau, delay):
  """
  Defines the general form of the ODE that governs the dynamics of the mass
  flow controllers.
  
  Args:
    t (double) time input needed by ODE solver
    y (double array) ODE state [y, y_dot, y_ddot]
    u (double) input to the ode
    tau (double) time constant of the first-order ode with delay
    delay (double) time delay term
    
  Returns:
    [y_dot, y_ddot, y_dddot] array of time-derivatives of state vector
      evaluated using a second-order Pade approximation of the time-delayed
      first-order ODE dydt = K*u(t-delay)*heavyside(t-delay)/tau - y(t)/tau
  """
  
  den = tau*delay**2
  # dydt0 = y[1]
  # dydt1 = y[2]
  dydt2 = -12*y[0]/den - y[1]*(6*delay + 12*tau)/den \
          - y[2]*(6*tau + delay)/tau/delay + 12*u/den
  return [y[1], y[2], dydt2]  # dydt array

def first_order_output(y, K, delay):
  """
  Defines the output function for the 2nd-order Pade approximation of a
  1st-order ODE with time delay. Used to get pressure from the state equation.
  
  Args:
    y (list, double) current state of the ODE
    K (double) gain of the first-order ODE
    delay (double) time delay term
  
  Returns:
    P (double) pressure, approximation of first-order ODE response with time
      delay
  """
  
  return y[0]*K - y[1]*K*delay/2 + y[2]*K*delay**2/12

class Instrument():
  """
  Defines the instrument class. All of the different measurement apparatus that
  we simulate will be children of this class.
  """
  def __init__(self, location):
    """
    Constructor
    
    Args:
      location double-valued location of the instrument from the combustor.
    """
    self.location = location
  
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
      p_atm (double) atmospheric pressure in psi
      t_step (double) time step for system simulation
    '''
    
    # Initialize constants
    self.p_atm = p_atm
    self.t_step = t_step
    self.time = 0.0  # initialize current time to zero
    
    # Initialize sensor list
    self.sensor_list = [StaticSensor(gain=10.0, offset=1.0, location=0.0),
                        DynamicSensor(tf=tf1, location=1.0)]
    #TODO initialize others
    #TODO figure out transfer functions!
    
    # Initilaize flame
    self.flame = Flame(operating_map=lambda p: one_sphere(p, radius=2.0),
                       tf=flame_tf)
    #TODO figure out operating space maps!
    
    # Initialize mass flow controller (MFC) list
    #TODO set these!
    K_mfcs = [10.0, 5.0, 2.0, 1.0]  # MFC gains
    tau_mfcs = [1.5, 2.5, 3.0, 1.4]  # MFC time constants
    td_mfcs = [0.2, 0.1, 0.4, 0.3]  # MFC time delays
    y0 = [0.0, 0.0, 0.0]  # initial value
    mfc_list = []
    
    for K, tau, td in K_mfcs, tau_mfcs, td_mfcs:
      mfc_list.append(MFC(lambda t, y, u: first_order_delay(t, y, u, tau, td),
                          lambda y: first_order_output(y, K, td), y0))
      
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
    
    flame_snapshot = self.flame.update(self.controller.mfc_list)
    #TODO figure out what the flame has to return
    
    for sensor in self.sensor_list:
      sensor.update(flame_snapshot, self.t_step)
    
    self.time += self.t_step

class Flame():
  """Defines the flame held within the combustor."""
  
  def __init__(self, operating_map, tf):
    """
    Constructor
    
    Args:
      map (function) maps out the actual stable/unstable regions of the
        system's operating space.
      tf (undecided) transfer function of the flame.
    """
    
    self.operating_map = operating_map
    self.tf = tf
    self.state = False
    
  def ignite(self):
    """Lights the flame by setting its state to True."""
    
    self.state = True
  
  def blowout(self):
    """Extinguishes the flame by setting its state to False."""
    
    self.state = False
    
  def update(self, mfc_list):
    """
    Updates the state of flame and outputs appropriate stuff.
    
    Args:
      mfc_list (list) MFC objects that are used to determine the flame state.
    
    Returns:
      flame_snapshot (undecided) I am not sure what this should be yet.
    """
    
    # Build the operating point from the MFC list
    operating_point = []
    for mfc in mfc_list:
      operating_point.append(mfc.get_output()[0])
    
    #TODO update the flame model and return some snapshot of the physical
    # combustor space maybe?
    
    # Update the flame state based on operating map
    if not self.state and self.operating_map(operating_point):
      self.ignite()
    if self.state and not self.operating_map(operating_point):
      self.blowout()
      
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
    """Return the internal time used by the MFC ODE."""
    
    return self.ode.t
    
  def get_output(self):
    """Return the current output of the MFC through its output function."""
    
    return self.output_fcn(self.ode.y)
  
  def get_state(self):
    """Return the current internal state of the MFC ODE."""
    
    return self.ode.y
    
  def update(self, input_val, t_step):
    """
    Updates the output of the MFC step-by-step.
    
    Args:
      input (double) controller input to the MFC ODE (volts).
      t_step (double) time step used for simulation.
    
    Returns:
      mass_flow (double) mass flow rate (LPM or something).
    """
    
    self.ode.set_f_params(input_val)
    self.ode.integrate(self.ode.t + t_step)
    return self.ode.successful()

class StaticSensor(Instrument):
  """Defines a static sensor object."""
  
  def __init__(self, gain, offset, location):
    """
    Constructor
    
    Args:
      gain (double) static gain of the sensor.
      offset (double) constant offset of the sensor.
      location (double) location of the sensor from the combustor.
    """
    
    self.gain = gain
    self.offset = offset
    self.state = 0.0
    super(StaticSensor, self).__init__(location)  # parent keeps location
    # should be able to access location with self.location!
    
  def get_output(self):
    """Return the state of the sensor."""
    
    return self.state
  
  def update(self, in_value, t_step):
    """
    Update the output of the static sensor.
    
    Args:
      input I don't know what this should be yet.
      t_step (double) time step for simulation.
    """
    
    self.state = in_value*self.gain + self.offset

class DynamicSensor(Instrument):
  """Defines the dynamic sensor object."""
  
  def __init__(self, tf, location):
    """
    Constructor
    
    Args:
      tf (undecided) transfer function defining the response of the sensor to stimulus.
      location (double) location of the sensor from the combustor.
    """
    
    self.tf = tf
    self.state = 0.0
    super(DynamicSensor, self).__init__(location)
  
  def get_output(self):
    """Return the state of the sensor."""
    
    return self.state
  
  def output_time_series(self):
    """Output the response of the sensor as a time series."""
    
    pass
  
  def output_power_spectrum(self):
    """Output the response of the sensor as a power spectrum."""
    
    pass
  
  def update(self, in_value, t_step):
    """
    Update the response of the sensor.
    
    Args:
      input I don't know what this should be yet.
      t_step (double) time step for simulation.
    """
    
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
      mfc_list (list of MFC) MFC objects defining the mass flow controllers.
      control_law_list (list of functions) control laws that apply to each MFC
        in the mfc_list
      t_step_ctrl (double) iteration rate for the control law. Should be longer
        than the simulation time step.
    """
    
    self.t_step_ctrl = t_step_ctrl  # time step for updating the control law
    self.mfc_list = mfc_list
    self.control_law_list = control_law_list
    self.u_ctrl = [0]*len(mfc_list)
  
  def control_law(self, mass_flow_des):
    """
    Determines the control effort given the desired state, the current state,
    and the controller time step.
    
    Args:
      mass_flow_des (list, double) desired mass flow rates for MFCs (LPM)
    """
    
    #TODO test for length of mass_flow_desired
    u = []
    for mfc, ctrl_law, ref in self.mfc_list, self.control_law_list, mass_flow_des:
      u.append(ctrl_law(mfc.get_output(), ref))  # prop. ctrl. only
    self.u_ctrl = u
    
  def update(self, mass_flow_des, t_step, time):
    """
    Update the controller and its MFCs.
    
    Args:
      mass_flow_des (double array) desired mass flow rates for MFCs (LPM)
      t_step (double) time step used for simulation (seconds)
      time (double) current simulation time (seconds)
    """
    
    # Update the control effort based on the current and desired state when the
    # simulation time reaches another controller period.
    #TODO check that this works!
    if time % self.t_step_ctrl < 1.0:
      self.control_law(mass_flow_des)
    
    #TODO figure out how to derive a control effort from a control law.
    for mfc, u in self.mfc_list, self.u_ctrl:
      if not mfc.update(u, t_step):  # run the update and check that it completed successfully
        return False
      else:
        return True