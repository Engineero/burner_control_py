'''
Created on Jan 8, 2016

@author: nathantoner
'''

import unittest
import matplotlib.pyplot as plt
import numpy as np
from burner_control import lab_hardware, sim_functions

plot_flag = False  # True if we want some tests to plot results


def test_ode(t, y, u, K, tau):
  """
  Defines a first-order ODE for testing the MFC update method.
      
  Args:
    t (double): time required by odeint
    y (double): array of current ODE state
    u (double): input to the ODE
    K (double): first-order system gain
    tau (double): first-order system time constant
    
  Returns:
    double: response = K*u/tau - y/tau
  """
      
  return K*u/tau - y/tau

def test_ctrl_law(y, ref, K):
  """
  Defines a simple proportional control law.
  
  Args:
    y (double): current state of an MFC
    ref (double): desired state of the MFC
    K (double): proportional controller gain
  
  Returns:
    double: control effort u = K*(ref - y)
  """
  
  return K*(ref - y)

class TestLabHardware(unittest.TestCase):
  """Unit tests for classes in lab_hardware.py."""

  def test_mfc_class(self):
    """Tests the lab_hardware.MFC class."""
    
    # Initialize constants and lists used for tests
    K = 10.0
    tau = 1.0
    td = 0.2
    y0_1 = [0.0]
    y0_2 = [0.0]*3
    time = 0.0
    t_list = []
    t_step = 0.01
    stop_time = 10.0
    input_val = 1.0
    response = []
    
    # Define model matrices
    den = tau*td**2
    A = np.array([[0, 1, 0], [0, 0, 1], [-12/den, -(6*td + 12*tau)/den, -(6*tau + td)/tau/td]])
    B = np.array([[0], [0], [12/den]])
    C = np.array([K, -K*td/2, K*td**2/12])
    
    # Initialize the MFCs
    test_mfc1 = lab_hardware.MFC(lambda t, y, u: test_ode(t, y, u, K, tau),
                                 lambda x: x[0], y0_1)
    test_mfc2 = lab_hardware.MFC(lambda t, y, u: sim_functions.first_order_delay(t, y, u, A, B),
                                 lambda y: sim_functions.first_order_output(y, C),
                                 y0_2)
    
    # Run initialization tests on the MFC.
    self.assertIsInstance(test_mfc1, lab_hardware.MFC,
                          "Failure to initialize MFC1 to MFC class.")
    self.assertIsInstance(test_mfc1.get_output(), float,
                          "State of MFC1 is not returned as a float")
    self.assertIsInstance(test_mfc2.get_output(), np.ndarray,
                          "State of MFC2 is not returned as np.ndarray")
    self.assertEqual(test_mfc1.get_output(), y0_1[0],
                     "MFC1 state failed to initialize to y0_1")
    self.assertEqual(test_mfc2.get_output(), y0_2[0],
                     "MFC2 state failed to initialize to y0_2")
    self.assertEqual(test_mfc1.get_time(), 0.0,
                     "MFC1 time failed to initialize to 0.0")
    
    # Run the MFC a bit
    print("MFC1 output\tMFC2 output")
    while test_mfc2.get_time() < stop_time:
      if test_mfc1.update(input_val, t_step) and test_mfc2.update(input_val, t_step):
        t_list.append(time)
        response.append([test_mfc1.get_output(), test_mfc2.get_output()])
        if time == 0.0 or test_mfc2.get_time() % 1.0 < t_step:
          print("{}\t{}".format(test_mfc1.get_output(), test_mfc2.get_output()))
        time += t_step
      else:
        break
    
    # Test the MFC after running
    self.assertAlmostEqual(test_mfc1.get_output(), K, places=3,
                           msg="Final value of MFC1 not close to expected value")
    self.assertEqual(test_mfc1.get_time(), time,
                     "MFC1 simulation time out of sync with global simulation time")
    self.assertAlmostEqual(test_mfc1.get_time(), stop_time, delta=t_step,
                           msg="MFC1 simulation time not equal to stop time.")
    
    # Test the second MFC in simulation
    self.assertTrue(test_mfc2.get_output() != y0_2[0],
                    "MFC state failed to update from y0")
    self.assertAlmostEqual(test_mfc2.get_output()[0], K, places=2,
                           msg="Final value of MFC2 not close to expected value")
    self.assertEqual(test_mfc2.get_time(), time,
                     "MFC2 simulation time out of sync with global simulation time")
    self.assertAlmostEqual(test_mfc2.get_time(), stop_time, delta=t_step,
                           msg="MFC2 simulation time not equal to stop time.")
    
      # Plot the response of the MFCs
    if plot_flag:
      plt.plot(t_list, response)
      plt.grid(True)
      plt.xlabel("Time (seconds)")
      plt.ylabel("MFC Response (LPM)")
      plt.title("Unit Step Response of First-Order MFC Simulation")
      plt.legend(["First Order", "Pade Approx. w/ Delay"])
      plt.show()  # called at end to prevent plot from automatically closing

  def test_controller_class(self):
    """Tests the lab_hardware.Controller class."""
    
    # Initialize constants and lists used for test
    K_list = [10.0, 5.0]
    tau_list = [2.0, 1.0]
    y0 = [0]
    t_step_ctrl = 0.1  # control update rate (seconds)
    Kp_list = [4.0, 5.0]
    mass_flow_des = [4.0, 2.0]  # desired flow rates per MFC
    expected = [Kp*K*A/(1 + Kp*K) for Kp, K, A in zip(Kp_list, K_list, mass_flow_des)]
    time = 0.0
    t_step = 0.01
    t_list = []
    stop_time = 10.0
    response = []
    
    # Initialize list of MFCs
    mfc_list = [lab_hardware.MFC(lambda t, y, u: test_ode(t, y, u, K, tau),
                                 lambda x: x[0], y0)
                for K, tau in zip(K_list, tau_list)]
    
    # Initialize list of control laws
    control_law_list = [lambda y, ref: test_ctrl_law(y, ref, K) for K
                        in Kp_list]
    
    # Initialize controller object with one MFC
    test_ctrl = lab_hardware.Controller(mfc_list[0], control_law_list[0],
                                        t_step_ctrl)
    
    # Test the single-MFC controller
    #TODO test more fringe cases for initialization, add checks and exceptions to classes
    self.assertIsInstance(test_ctrl, lab_hardware.Controller,
                          "Single MFC controller not initialized to correct class.")
    self.assertEqual(test_ctrl.get_time(), time,
                     "Single MFC controller initial time not equal to initial simulation time.")
    self.assertIsInstance(test_ctrl.get_output(), list,
                          "Single MFC controller output not returned as list.")
    self.assertListEqual(test_ctrl.get_output(), y0,
                         "Single MFC controller initial value not equal to y0")
    
    # Initialize controller object with multiple MFCs
    test_ctrl = lab_hardware.Controller(mfc_list, control_law_list, t_step_ctrl)
    
    # Test initialization
    self.assertIsInstance(test_ctrl, lab_hardware.Controller,
                          "Multiple MFC controller not initialized to correct class.")
    self.assertEqual(test_ctrl.get_time(), time,
                     "Multiple MFC controller initial time not equal to initial simulation time.")
    self.assertIsInstance(test_ctrl.get_output(), list,
                          "Multiple MFC controller output not returned as list.")
    self.assertListEqual(test_ctrl.get_output(), [0.0, 0.0],
                         "Multiple MFC controller initial value not equal to y0")
    
    # Run the controller
    print("Time\tControlled Response")
    while time < stop_time:
      if test_ctrl.update(mass_flow_des, t_step):
        t_list.append(time)
        response.append(test_ctrl.get_output())
        if time == 0.0 or time % 1.0 < t_step:
          print("{}\t{}".format(time, test_ctrl.get_output()))
        time += t_step
      else:
        break
    
    #TODO test the result of running the controller
    self.assertAlmostEqual(test_ctrl.get_time(), stop_time, delta=t_step,
                           msg="Controller's simulation time not equal to global stop time.")
    for res, exp in zip(test_ctrl.get_output(), expected):
      self.assertAlmostEqual(res, exp, delta=0.1,
                       msg="Controlled result not equal to expected result.")
    
    # PLot the response of the controlled MFCs
    if plot_flag:
      plt.plot(t_list, response)
      plt.grid(True)
      plt.xlabel("Time (seconds)")
      plt.ylabel("MFC Responses (LPM)")
      plt.title("Controlled Response of MFCs")
      plt.legend(["MFC1 r = {}".format(mass_flow_des[0]),
                  "MFC2 r = {}".format(mass_flow_des[1])])
      plt.show()  # called at end to prevent plot from automatically closing
    
  def test_flame_class(self):
    """Tests for the lab_hardware.Flame class."""
    
    # Initialize constants
    map_radius = 1.0
    good_points = [[2.0, 2.0],
                   [1.0001, 0.0],
                   [100, 200],
                   [0.0, 1.0001],
                   [-1.0001, 0.0],
                   [0.0, -1.0001],
                   [-2.0, -2.0],
                   [2.0, 0, 0.1, -0.1]]
    bad_points = [[1.0, 0.0],
                  [0.0, 1.0],
                  [0.0, 0.0],
                  [0.5, 0.5],
                  [-1.0, 0.0],
                  [0.0, -1.0],
                  [-0.5, -0.5],
                  [0.1, 0.5, -0.5, 0.1]]
    
    # Initialize flame object
    test_flame = lab_hardware.Flame(lambda p: sim_functions.one_sphere(p, map_radius))
    
    # Test the initialization
    self.assertIsInstance(test_flame, lab_hardware.Flame,
                          "Test flame object not initialized to Flame class.")
    self.assertFalse(test_flame.get_state(),
                     "Flame state not initialized to False")
    
    # Test the flame ignites when given a valid operating point
    for point in good_points:
      self.assertEqual(test_flame.update(point), True,
                       "Flame not returning correct state from update method (True).")
      self.assertTrue(test_flame.get_state(),
                      "Flame does not ignite when moved to good op. point.")
    
    # Test the flame blows out when given a bad operating point
    for point in bad_points:
      self.assertEqual(test_flame.update(point), False,
                       "Flame not returning correct state from update method (False).")
      self.assertFalse(test_flame.get_state(),
                       "Flame does not blow out when moved to a bad op. point.")

  def test_static_sensor_class(self):
    """Tests for the lab_hardware.StaticSensor class."""
    
    # Initialize constants
    location = 1.0
    K = 2.0  # sensor gain
    offset = 0.5  # sensor offset
    mean = 0.0  # Gaussian white noise mean
    std = 0.01  # Gaussian white noise standard deviation
    P_list = [0, 1, 2, 3, 4, 5, 14.0, -1.1, -2]  # actual pressures to read
    
    # Initialize StaticSensor object
    test_sensor = lab_hardware.StaticSensor(lambda y: sim_functions.static_model(y, K, offset, mean, std),
                                            location)
    
    # Test initialization
    self.assertIsInstance(test_sensor, lab_hardware.StaticSensor,
                          "Test static sensor not initialized to StaticSensor class.")
    self.assertEqual(test_sensor.get_output(), 0.0,
                     "Sensor does not return expected initial reading.")
    self.assertEqual(test_sensor.get_location(), location,
                     "Sensor object does not return the correct location.")
    
    # Test readings with the sensor
    for P_actual in P_list:
      expected = sim_functions.static_model(P_actual, K, offset)
      self.assertNotEqual(test_sensor.update(P_actual), expected,
                          "Measured pressure with noise equals value without noise. Noise may not be working.")
      self.assertAlmostEqual(test_sensor.get_output(), expected,
                             delta=4*std,
                             msg="Stored sensor state >3*std away from expected state.")

  def test_kalman_filter_class(self):
    """Tests for the lab_hardware.KalmanFilter class."""
    
    # Define constants
    K = 10.0
    tau = 1.0
    td = 0.2
    den = tau*td**2
    A = np.array([[0, 1, 0], [0, 0, 1], [-12/den, -(6*td + 12*tau)/den, -(6*tau + td)/tau/td]])
    B = np.array([[0], [0], [12/den]])
    C = np.array([K, -K*td/2, K*td**2/12])
    Q = 1e-5*np.identity(len(B))  # process noise covariance
    R = 1e-2  # measurement noise covariance
    P = Q  # initial estimate of error covariance
    
    # Initialize the KF and system ODE
    test_KF = lab_hardware.KalmanFilter(A, B, C, Q, R, P)
    
    # Test initialization
    self.assertIsInstance(test_KF, lab_hardware.KalmanFilter,
                          "Kalman filter not initialized to KF class.")
    self.assertListEqual(test_KF.get_output().tolist(),
                         np.zeros(A.shape[1]).tolist(),
                     "Kalman filter initial state does not match expected value.")
    self.assertListEqual(test_KF.get_err_cov().tolist(), P.tolist(),
                     "Kalman filter initial estimated error covariance does not match expected.")
    
    # Test methods
    self.assertIsInstance(test_KF.get_output(), np.ndarray,
                          "Kalman filter get_output method does not return expected data type.")
    self.assertIsInstance(test_KF.get_err_cov(), np.ndarray,
                          "Kalman filter get_err_cov method does not return expected data type")
    self.assertIsInstance(test_KF.update([0], [0]), np.ndarray,
                          "Kalman filter update method does not return expected data type.")

    
if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testLabHardware']
  unittest.main()