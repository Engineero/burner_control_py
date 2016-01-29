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
    t (float): time required by odeint
    y (float): array of current ODE state
    u (float): input to the ODE
    K (float): first-order system gain
    tau (float): first-order system time constant
    
  Returns:
    float: response = K*u/tau - y/tau
  """
      
  return K*u/tau - y/tau

def test_ctrl_law(e, K):
  """
  Defines a simple proportional control law.
  
  Args:
    e (float): current state error (ref-x) of an MFC
    K (float): proportional controller gain
  
  Returns:
    float: control effort u = K*(ref - y)
  """
  
  return K*e

class TestLabHardware(unittest.TestCase):
  """Unit tests for classes in lab_hardware.py."""
  
  def setUp(self):
    """Run repeated setup code once."""
    
    # Define class and local parameters
    unittest.TestCase.setUp(self)
    self.K_list = [3, 10]
    self.tau_list = [0.3, 1]
    self.delay_list = [0.03, 0.1, 0.3]
    self.t_step = 0.01
    self.stop_time = 10.0
    self.input_val = 1.0
    self.t_step_ctrl = 0.1
    self.y0_1 = [0]
    self.y0_2 = [0]*3
    A, B, self.C = sim_functions.get_state_matrices(self.K_list[-1],
                                               self.tau_list[-1],
                                               self.delay_list[0])
    self.Kp_list = [4.0, 5.0]
    self.mass_flow_des = [4.0, 2.0]
    self.std = 1.0
    self.Ks = 2.0
    Q = 1e-5*np.identity(A.shape[0])  # process noise covariance
    R = self.std  # measurement noise covariance
    self.P = Q  # initial estimate of error covariance
    self.offset = 0.0
    mean = 0.0
    
    # Define objects
    self.test_mfc1 = lab_hardware.MFC(lambda t, y, u: test_ode(t, y, u, self.K_list[-1], self.tau_list[-1]),
                                      lambda x: x[0], self.y0_1)
    self.test_mfc2 = lab_hardware.MFC(lambda t, y, u: sim_functions.first_order_delay(t, y, u, A, B),
                                      lambda y: sim_functions.first_order_output(y, self.C),
                                      self.y0_2)
    self.mfc_list = [lab_hardware.MFC(lambda t, y, u: test_ode(t, y, u, K, tau),
                                      lambda x: x[0], self.y0_1)
                     for K, tau in zip(self.K_list, self.tau_list)]
    self.test_KF = lab_hardware.KalmanFilter(A, B, self.Ks*self.C, Q, R, self.P)
    self.test_sensor = lab_hardware.StaticSensor(lambda y: sim_functions.static_model(y, self.Ks, self.offset, mean, self.std),
                                                 1.0)

  def test_mfc_class(self):
    """Tests the lab_hardware.MFC class."""
    
    # Initialize constants and lists used for tests
    time = 0.0
    t_list = []
    response = []
    
    # Run initialization tests on the MFC.
    self.assertIsInstance(self.test_mfc1, lab_hardware.MFC,
                          "Failure to initialize MFC1 to MFC class.")
    self.assertIsInstance(self.test_mfc1.get_output(), float,
                          "State of MFC1 is not returned as a float")
    self.assertIsInstance(self.test_mfc2.get_output(), np.ndarray,
                          "State of MFC2 is not returned as np.ndarray")
    self.assertEqual(self.test_mfc1.get_output(), self.y0_1[0],
                     "MFC1 state failed to initialize to y0_1")
    self.assertEqual(self.test_mfc2.get_output(), self.y0_2[0],
                     "MFC2 state failed to initialize to y0_2")
    self.assertEqual(self.test_mfc1.get_time(), 0.0,
                     "MFC1 time failed to initialize to 0.0")
    
    # Run the MFC a bit
    print("MFC1 output\tMFC2 output")
    while self.test_mfc2.get_time() < self.stop_time:
      if self.test_mfc1.update(self.input_val, self.t_step) and \
      self.test_mfc2.update(self.input_val, self.t_step):
        t_list.append(time)
        response.append([self.test_mfc1.get_output(),
                         self.test_mfc2.get_output()])
        if time == 0.0 or self.test_mfc2.get_time() % 1.0 < self.t_step:
          print("{}\t{}".format(self.test_mfc1.get_output(),
                                self.test_mfc2.get_output()))
        time += self.t_step
      else:
        break
    
    # Test the MFC after running
    self.assertAlmostEqual(self.test_mfc1.get_output(), self.K_list[-1],
                           places=3,
                           msg="Final value of MFC1 not close to expected value")
    self.assertEqual(self.test_mfc1.get_time(), time,
                     "MFC1 simulation time out of sync with global simulation time")
    self.assertAlmostEqual(self.test_mfc1.get_time(), self.stop_time,
                           delta=self.t_step,
                           msg="MFC1 simulation time not equal to stop time.")
    
    # Test the second MFC in simulation
    self.assertTrue(self.test_mfc2.get_output() != self.y0_2[0],
                    "MFC state failed to update from y0")
    self.assertAlmostEqual(self.test_mfc2.get_output()[0][0], self.K_list[-1],
                           places=2,
                           msg="Final value of MFC2 not close to expected value")
    self.assertEqual(self.test_mfc2.get_time(), time,
                     "MFC2 simulation time out of sync with global simulation time")
    self.assertAlmostEqual(self.test_mfc2.get_time(), self.stop_time,
                           delta=self.t_step,
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
    expected = [Kp*K*A/(1 + Kp*K) for Kp, K, A in
                zip(self.Kp_list, self.K_list, self.mass_flow_des)]
    time = 0.0
    t_list = []
    response = []
    
    # Initialize list of control laws
    control_law_list = [lambda e: test_ctrl_law(e, K) for K in self.Kp_list]
    
    # Initialize controller object with one MFC
    test_ctrl = lab_hardware.Controller(self.mfc_list[0], control_law_list[0],
                                        self.t_step_ctrl)
    
    # Test the single-MFC controller
    #TODO test more fringe cases for initialization, add checks and exceptions to classes
    self.assertIsInstance(test_ctrl, lab_hardware.Controller,
                          "Single MFC controller not initialized to correct class.")
    self.assertEqual(test_ctrl.get_time(), time,
                     "Single MFC controller initial time not equal to initial simulation time.")
    self.assertIsInstance(test_ctrl.get_output(), list,
                          "Single MFC controller output not returned as list.")
    self.assertListEqual(test_ctrl.get_output(), self.y0_1,
                         "Single MFC controller initial value not equal to y0")
    
    # Initialize controller object with multiple MFCs
    test_ctrl = lab_hardware.Controller(self.mfc_list, control_law_list,
                                        self.t_step_ctrl)
    
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
    while time < self.stop_time:
      if test_ctrl.update(self.mass_flow_des, self.t_step):
        t_list.append(time)
        response.append(test_ctrl.get_output())
        if time == 0.0 or time % 1.0 < self.t_step:
          print("{}\t{}".format(time, test_ctrl.get_output()))
        time += self.t_step
      else:
        break
    
    #TODO test the result of running the controller
    self.assertAlmostEqual(test_ctrl.get_time(), self.stop_time,
                           delta=self.t_step,
                           msg="Controller's simulation time not equal to global stop time.")
    for res, exp in zip(test_ctrl.get_output(), expected):
      self.assertAlmostEqual(res, exp, delta=0.1*exp,
                       msg="Controlled result not within 10% of expected result.")
    
    # PLot the response of the controlled MFCs
    if plot_flag:
      plt.plot(t_list, response)
      plt.grid(True)
      plt.xlabel("Time (seconds)")
      plt.ylabel("MFC Responses (LPM)")
      plt.title("Controlled Response of MFCs")
      plt.legend(["MFC1 r = {}".format(self.mass_flow_des[0]),
                  "MFC2 r = {}".format(self.mass_flow_des[1])])
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
    P_list = [0, 1, 2, 3, 4, 5, 14.0, -1.1, -2]  # actual pressures to read
    
    # Test initialization
    self.assertIsInstance(self.test_sensor, lab_hardware.StaticSensor,
                          "Test static sensor not initialized to StaticSensor class.")
    self.assertEqual(self.test_sensor.get_output(), 0.0,
                     "Sensor does not return expected initial reading.")
    self.assertEqual(self.test_sensor.get_location(), 1.0,
                     "Sensor object does not return the correct location.")
    
    # Test readings with the sensor
    for P_actual in P_list:
      expected = sim_functions.static_model(P_actual, self.Ks, self.offset)
      self.assertNotEqual(self.test_sensor.update(P_actual), expected,
                          "Measured pressure with noise equals value without noise. Noise may not be working.")
      self.assertAlmostEqual(self.test_sensor.get_output(), expected,
                             delta=4*self.std,
                             msg="Stored sensor state >3*std away from expected state.")

  def test_kalman_filter_class(self):
    """Tests for the lab_hardware.KalmanFilter class."""
    
    # Define constants
    t_list = []
    time = 0
    response = []

    # Test initialization
    self.assertIsInstance(self.test_KF, lab_hardware.KalmanFilter,
                          "Kalman filter not initialized to KF class.")
    self.assertListEqual(self.test_KF.get_output().flatten().tolist(),
                         self.y0_2,
                     "Kalman filter initial state does not match expected value.")
    self.assertListEqual(self.test_KF.get_err_cov().tolist(), self.P.tolist(),
                     "Kalman filter initial estimated error covariance does not match expected.")
    
    # Test method return types
    self.assertIsInstance(self.test_KF.get_output(), np.ndarray,
                          "Kalman filter get_output method does not return expected data type.")
    self.assertIsInstance(self.test_KF.get_err_cov(), np.ndarray,
                          "Kalman filter get_err_cov method does not return expected data type")
    
    # Run the MFC a bit with sensor
    print("Time\tMFC output\tMFC output with noise\tKalman filter output")
    while self.test_mfc2.get_time() < self.stop_time:
      if self.test_mfc2.update(self.input_val, self.t_step):
        t_list.append(time)
        reading = self.test_sensor.update(self.test_mfc2.get_output())
        KF_value = self.test_KF.update(reading, self.input_val)
        response.append([self.test_mfc2.get_output()[0][0], reading[0][0],
                         self.C.dot(KF_value)[0][0]])
        if time == 0.0 or self.test_mfc2.get_time() % 1.0 < self.t_step:
          print("{}\t{}".format(self.test_mfc2.get_time(),
                                response[-1]))
        time += self.t_step
      else:
        break
    
    # Test the result of running the system with the KF
    self.assertAlmostEqual(response[-1][0], self.K_list[-1]*self.input_val,
                           delta=3*self.std,
                           msg="Filtered response value not within 3*std of expected value.")
    
    # Plot the response of the MFCs
    if plot_flag:
      plt.plot(t_list, response)
      plt.grid(True)
      plt.xlabel("Time (seconds)")
      plt.ylabel("MFC Response (LPM)")
      plt.title("Unit Step Response of First-Order MFC Simulation with Noise and Kalman Filter")
      plt.legend(["Pade Approx. Sys.", "With Noise", "Kalman-Filtered"])
      plt.show()  # called at end to prevent plot from automatically closing

    
if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testLabHardware']
  unittest.main()