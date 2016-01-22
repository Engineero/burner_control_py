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
        point (list, double): list representing test condition
        radius (double, default=1.0): radius of hypersphere to test
    
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
  
def first_order_delay(t, y, u, tau, delay):
  """
  Defines the general form of the ODE that governs the dynamics of the mass
  flow controllers.
  
  Args:
    t (double): time input needed by ODE solver
    y (double array): ODE state [y, y_dot, y_ddot]
    u (double): input to the ode
    tau (double): time constant of the first-order ode with delay
    delay (double): time delay term
    
  Returns:
    list: [y_dot, y_ddot, y_dddot], time-derivatives of state vector evaluated
      using a second-order Pade approximation of the time-delayed first-order
      ODE dydt = K*u(t-delay)*heavyside(t-delay)/tau - y(t)/tau
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
    y (list, double): current state of the ODE
    K (double): gain of the first-order ODE
    delay (double): time delay term
  
  Returns:
    double: pressure, approximation of first-order ODE response with time delay
  """
  
  return y[0]*K - y[1]*K*delay/2 + y[2]*K*delay**2/12

def static_model(y, K, offset, mean=0, std=0):
  """
  Models the static pressure measurement sensor.
  
  Args:
    y (double): input to the static sensor
    K (double): static gain of the sensor
    offset (double): constant offset of the sensor
  
  Kwargs:
    mean (double, default=0): mean of Gaussian white noise
    std (double, default=0): standard deviation of Gaussian white noise. Set
      std = 0 for no noise.
    
  Returns:
    double: static reading = K*y + offset + noise
  """
  
  if std == 0:
    noise = 0
  else:
    noise = np.random.normal(mean, std)
  return K*y + offset + noise