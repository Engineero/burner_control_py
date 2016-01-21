'''
Created on Jan 8, 2016

@author: nathantoner
'''
import unittest
import matplotlib.pyplot as plt
from burner_control import lab_hardware

def test_ode(t, y, u, K, tau):
  """
  Defines a first-order ODE for testing the MFC update method.
      
  Args:
    t (double) time required by odeint
    y (double) array of current ODE state
    u (double) input to the ODE
    K (double) first-order system gain
    tau (double) first-order system time constant
    
  Returns:
    response (double) = K*u/tau - y/tau
  """
      
  return K*u/tau - y/tau

def test_ctrl_law(y, ref, K):
  """
  Defines a simple proportional control law.
  
  Args:
    y (double) current state of an MFC
    ref (double) desired state of the MFC
    K (double) proportional controller gain
  
  Returns:
    u (double) = K*(ref - y)
  """
  
  return K*(ref - y)

def test_static_model(y, K, offset):
  """
  Defines a simple static sensor model.
  
  Args:
    y (double) current input to the sensor
    K (double) static gain of the sensor
    offset (double) offset of the sensor
  
  Returns:
    reading (double) = K*y + offset
  """
  
  return K*y + offset

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
    
    # Initialize the MFCs
    test_mfc1 = lab_hardware.MFC(lambda t, y, u: test_ode(t, y, u, K, tau),
                                 lambda x: x[0], y0_1)
    test_mfc2 = lab_hardware.MFC(lambda t, y, u: lab_hardware.first_order_delay(t, y, u, tau, td),
                                 lambda y: lab_hardware.first_order_output(y, K, td),
                                 y0_2)
    
    # Run initialization tests on the MFC.
    self.assertIsInstance(test_mfc1, lab_hardware.MFC,
                          "Failure to initialize MFC1 to MFC class.")
    self.assertIsInstance(test_mfc1.get_output(), float,
                          "State of MFC1 is not returned as a float")
    self.assertIsInstance(test_mfc2.get_output(), float,
                          "State of MFC2 is not returned as float")
    self.assertEqual(test_mfc1.get_output(), y0_1[0],
                     "MFC1 state failed to initialize to y0_1")
    self.assertEqual(test_mfc2.get_output(), y0_2[0],
                     "MFC2 state failed to initialize to y0_2")
    self.assertEqual(test_mfc1.get_time(), 0.0,
                     "MFC1 time failed to initialize to 0.0")
    
    # Run the MFC a bit
    while test_mfc2.get_time() < stop_time:
      if test_mfc1.update(input_val, t_step) and test_mfc2.update(input_val, t_step):
        t_list.append(time)
        response.append([test_mfc1.get_output(), test_mfc2.get_output()])
        if time == 0.0 or test_mfc2.get_time() % 1.0 < t_step:
          print("{}\t{}".format(test_mfc1.get_output(), test_mfc2.get_output()))
        time += t_step
      else:
        break
    
    plt.plot(t_list, response)
    plt.grid(True)
    plt.xlabel("Time (seconds)")
    plt.ylabel("MFC Response (LPM)")
    plt.title("Unit Step Response of First-Order MFC Simulation")
    plt.legend(["First Order", "Pade Approx. w/ Delay"])
    plt.draw()  # draw() is non-blocking so test can continue
    
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
    self.assertAlmostEqual(test_mfc2.get_output(), K, places=2,
                           msg="Final value of MFC2 not close to expected value")
    self.assertEqual(test_mfc2.get_time(), time,
                     "MFC2 simulation time out of sync with global simulation time")
    self.assertAlmostEqual(test_mfc2.get_time(), stop_time, delta=t_step,
                           msg="MFC2 simulation time not equal to stop time.")
    
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
    while time < stop_time:
      if test_ctrl.update(mass_flow_des, t_step, time):
        t_list.append(time)
        response.append(test_ctrl.get_output())
        if time == 0.0 or time % 1.0 < t_step:
          print("{}\t{}".format(time, test_ctrl.get_output()))
        time += t_step
      else:
        break
    
    plt.plot(t_list, response)
    plt.grid(True)
    plt.xlabel("Time (seconds)")
    plt.ylabel("MFC Responses (LPM)")
    plt.title("Controlled Response of MFCs")
    plt.legend(["MFC1 r = {}".format(mass_flow_des[0]),
                "MFC2 r = {}".format(mass_flow_des[1])])
    plt.draw()  # draw() is non-blocking so test can continue
    
    #TODO test the result of running the controller
    self.assertAlmostEqual(test_ctrl.get_time(), stop_time, delta=t_step,
                           msg="Controller's simulation time not equal to global stop time.")
    for res, exp in zip(test_ctrl.get_output(), expected):
      self.assertAlmostEqual(res, exp, delta=0.1,
                       msg="Controlled result not equal to expected result.")
    
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
    test_flame = lab_hardware.Flame(lambda p: lab_hardware.one_sphere(p, map_radius))
    
    # Test the initialization
    self.assertIsInstance(test_flame, lab_hardware.Flame,
                          "Test flame object not initialized to Flame class.")
    self.assertFalse(test_flame.get_state(),
                     "Flame state not initialized to False")
    
    # Test the flame ignites when given a valid operating point
    for point in good_points:
      test_flame.update(point)
      self.assertTrue(test_flame.get_state(),
                      "Flame does not ignite when moved to good op. point.")
    
    # Test the flame blows out when given a bad operating point
    for point in bad_points:
      test_flame.update(point)
      self.assertFalse(test_flame.get_state(),
                       "Flame does not blow out when moved to a bad op. point.")

  def test_static_sensor_class(self):
    """Tests for the lab_hardware.StaticSensor class."""
    
    # Initialize constants
    location = 1.0
    K = 2.0  # sensor gain
    offset = 0.5  # sensor offset
    P_actual = 14.0  # actual pressure to read
    
    # Initialize StaticSensor object
    test_sensor = lab_hardware.StaticSensor(lambda y: test_static_model(y, K, offset),
                                            location)
    
    # Test initialization
    self.assertIsInstance(test_sensor, lab_hardware.StaticSensor,
                          "Test static sensor not initialized to StaticSensor class.")
    self.assertEqual(test_sensor.get_output(), 0.0,
                     "Sensor does not return expected initial reading.")
    
    # Test readings with the sensor
    self.assertEqual(test_sensor.update(P_actual), K*P_actual + offset,
                     "Measured pressure does not equal expected value.")
    self.assertEqual(test_sensor.get_output(), K*P_actual + offset,
                     "Stored sensor state does not equal expected value.")
    
    
if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testLabHardware']
  unittest.main()