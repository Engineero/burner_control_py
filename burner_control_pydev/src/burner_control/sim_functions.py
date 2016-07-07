'''
Created on Jan 22, 2016

@author: nathantoner
'''

import numpy as np
import scipy.linalg
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
    return 1 if point ** 2 > radius ** 2 else 0
  else:
    return 1 if sum([num ** 2 for num in point]) > radius ** 2 else 0
  
def system_state_update(t, y, u, A, B):
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
  
  # TODO add an option for noise to this model:
  # A.dot(y.reshape(len(y), 1)) + B.dot(u) + N.dot(np.random.normal(mean, std))
  # maybe use np.random.normal(mean, std, (len(y), 1)) to return an array of
  # noise, then define "N" with the appropriate dimensions.
  return A.dot(y.reshape(len(y), 1)) + B.dot(u)

def system_output(y, C, mean=0, std=0):
  """
  Defines the output function for the general ODE with time delay used to model
  the MFCs. Used to get pressure from the state equation.
  
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
  return K * y + offset + noise

def make_lqr_law(A, B, Q, R):
  """
  Generates a linear quadratic regulator (LQR) control law and returns the
  controller's gain matrix.
  
  Args:
    A (ndarray): state transition matrix for linear system
    B (ndarray): input matrix for the linear system
    Q (ndarray): state error weighing matrix
    R (ndarray): control effort weighing matrix
  
  Returns:
    ndarray: LQR control gain matrix, K: u = -K*x
    ndarray: discrete Ricatti matrix P
    ndarray: eigen-values of the system A - B*K
  """
  
  # Solve the discrete-time Ricatti equation
  P = np.array(scipy.linalg.solve_discrete_are(A, B, Q, R))
  
  # Compute the LQR gain
  K = np.array(scipy.linalg.inv(R).dot((B.T).dot(P)))  # continuous LQR
#   K = np.array(scipy.linalg.inv((B.T).dot(P).dot(B) + R)
#                .dot((B.T).dot(P).dot(A)))  # discrete LQR
  eigen_values, _ = scipy.linalg.eig(A - B.dot(K))
  
  return K, P, eigen_values

def get_state_matrices(K, tau, td):
  """
  Generates the state space equation matrices for a first-order system with
  second-order Pade approximation of time delay.
  
  Args:
    K (float): first-order system gain
    tau (float): first-order system time constant
    td (float): time delay, approximated by second-order Pade approximation
    
  Returns:
    A, B, C (ndarray): matrices of system state space equation:
      x_dot = A*x + B*u
      y = C*x
  """
  
  den = tau * td ** 2
  A = np.array([[0, 1, 0], [0, 0, 1], [-12 / den, -(6 * td + 12 * tau) / den, -(6 * tau + td) / tau / td]])
  B = np.array([[0], [0], [12 / den]])
  C = np.array([[K, -K * td / 2, K * td ** 2 / 12]])
  return A, B, C

def get_second_ord_matrices(K, zeta, wn, td):
  """
  Generates the state space equation matrices for a second-order system with
  3,2-order Pade approximation of time delay.
  
  Args:
    K (float): second-order system gain
    zeta (float): second-order system damping ratio
    wn (float): second-order system natural frequency
    td (float): time delay, approximated by 3,2-order Pade approximation
    
  Returns:
    A, B, C (ndarray): matrices of system state space equation:
      x_dot = A*x + B*u
      y = C*x
  """
  
  td2 = td ** 2
  wn2 = wn ** 2
  A = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-20 * wn2 / td2, (-8 * td * wn2 - 40 * zeta * wn) / td2, (-td2 * wn2 - 40 * zeta * wn * td - 20) / td2, (-2 * zeta * td * wn - 8) / td]])
  B = np.array([[0], [0], [0], [20 * wn2 / td2]])
  C = np.array([[K, -K * 3 * td / 5, K * 3 * td2 / 20, -td ** 3 / 60]])
  return A, B, C
