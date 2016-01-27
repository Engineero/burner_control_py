'''
Created on Jan 22, 2016

@author: nathantoner
'''

import numpy as np
import numbers


def one_sphere(point, radius=1.0):
    """
    Checks whether points are inside or outside of a hypersphere.
    
    Args:
        point (list, float): list representing test condition
        radius (float, default=1.0): radius of hypersphere to test
    
    Returns:
        int: 1 if point is outside of the sphere (good), 0 if point is inside
          of or on the sphere (bad)
    
    Raises:
      ValueError: if radius <= 0
      TypeError: if input values are non-numeric or radius is a list
    """
    
    if radius <= 0:
      raise ValueError("Radius must be > 0.")
    
    if isinstance(point, numbers.Number):
      return 1 if point**2 > radius**2 else 0
    else:
      return 1 if sum([num**2 for num in point]) > radius**2 else 0
  
def first_order_delay(t, y, u, A, B):
  """
  Defines the general form of the ODE that governs the dynamics of the mass
  flow controllers.
  
  Args:
    t (float): time input needed by ODE solver
    y (ndarray): ODE state [y, y_dot, y_ddot]
    u (float): input to the single-input ode
    A (ndarray): state transition matrix for linear system
    B (ndarray): input matrix for the linear system
    
  Returns:
    ndarray: [y_dot, y_ddot, y_dddot], time-derivatives of state vector
      evaluated using a second-order Pade approximation of the time-delayed
      first-order ODE dydt = K*u(t-delay)*heavyside(t-delay)/tau - y(t)/tau
  """
  
  #TODO add an option for noise to this model:
  # A.dot(y.reshape(len(y), 1)) + B.dot(u) + N.dot(np.random.normal(mean, std))
  # maybe use np.random.normal(mean, std, (len(y), 1)) to return an array of
  # noise, then define "N" with the appropriate dimensions.
  return A.dot(y.reshape(len(y), 1)) + B.dot(u)

def first_order_output(y, C, mean=0, std=0):
  """
  Defines the output function for the 2nd-order Pade approximation of a
  1st-order ODE with time delay. Used to get pressure from the state equation.
  
  Args:
    y (ndarray): current state of the ODE
    C (ndarray): measurement matrix for the linear system
  
  Kwargs:
    mean (float, default=0): mean of Gaussian white measurement noise
    std (float, default=0): standard deviation of Gaussian white noise. Set
      std = 0 for no noise.
  
  Returns:
    float: pressure, approximation of first-order ODE response with time delay
      P = C*y
  """
  
#   return y[0]*K - y[1]*K*delay/2 + y[2]*K*delay**2/12
  if std == 0:
    noise = 0
  else:
    noise = np.random.normal(mean, std)
  return C.dot(y.reshape(len(y), 1)) + noise

def static_model(y, K, offset, mean=0, std=0):
  """
  Models the static pressure measurement sensor.
  
  Args:
    y (float): input to the static sensor
    K (float): static gain of the sensor
    offset (float): constant offset of the sensor
  
  Kwargs:
    mean (float, default=0): mean of Gaussian white noise
    std (float, default=0): standard deviation of Gaussian white noise. Set
      std = 0 for no noise.
    
  Returns:
    float: static reading = K*y + offset + noise
  """
  
  if std == 0:
    noise = 0
  else:
    noise = np.random.normal(mean, std)
  return K*y + offset + noise