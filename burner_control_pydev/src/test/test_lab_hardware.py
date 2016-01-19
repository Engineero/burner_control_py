'''
Created on Jan 8, 2016

@author: nathantoner
'''
import unittest
import matplotlib.pyplot as plt
from burner_control import lab_hardware

# Define an ODE that I know will work.
def test_ode(t, y, u, K, tau):
  """
  Defines a first-order ODE for testing the MFC update method.
      
  Args:
    t (double) time required by odeint
    y (double) array of current ODE state
    u (double) input to the ODE
    K (double) first-order system gain
    tau (double) first-order system time constant
  """
      
  return K*u/tau - y/tau

def test_ctrl_law(y, ref, K):
  """
  Defines a simple proportional control law.
  
  Args.
    y (double) current state of an MFC
    ref (double) desired state of the MFC
    K (double) proportional controller gain
  """
  
  return K*(y - ref)

class TestLabHardware(unittest.TestCase):
  """Unit tests for classes in lab_hardware.py."""

  def test_mfc_class(self):
    """
    Tests the MFC class, ensuring that it is initialized correctly, and that
    its methods work.
    """
    
    # Initialize constants and lists used for tests
    K = 10.0
    tau = 1.0
    td = 0.1
    y0_1 = [0.0]
    y0_2 = [0.0]*3
    time = 0.0
    t_list = []
    t_step = 0.01
    stop_time = 10.0
    input_val = 1.0
    response = []
    
    # Initialize the MFCs
    test_mfc1 = lab_hardware.MFC(lambda t, y, u: test_ode(t, y, u, K, tau), lambda x: x[0],
                                 y0_1)
    test_mfc2 = lab_hardware.MFC(lambda t, y, u: lab_hardware.first_order_delay(t, y, u, tau, td),
                                 lambda y: lab_hardware.first_order_output(y, K, td), y0_2)
    
    # Run initialization tests on the MFC.
    self.assertIsInstance(test_mfc1, lab_hardware.MFC,
                          "Failure to initialize MFC1 to MFC class.")
    self.assertIsInstance(test_mfc1.get_state(), float,
                          "State of MFC1 is not returned as a float")
    self.assertIsInstance(test_mfc2.get_state(), float,
                          "State of MFC2 is not returned as float")
    self.assertEqual(test_mfc1.get_state(), y0_1[0],
                     "MFC1 state failed to initialize to y0_1")
    self.assertEqual(test_mfc2.get_state(), y0_2[0],
                     "MFC2 state failed to initialize to y0_2")
    self.assertEqual(test_mfc1.get_time(), 0.0,
                     "MFC1 time failed to initialize to 0.0")
    
    # Run the MFC a bit
    while test_mfc2.get_time() < stop_time:
      if test_mfc1.update(input_val, t_step) and test_mfc2.update(input_val, t_step):
        t_list.append(time)
        response.append(test_mfc2.get_state())
        if time == 0.0 or test_mfc2.get_time() % 1.0 < t_step:
          print("{}\t{}".format(test_mfc1.get_state(), test_mfc2.get_state()))
        time += t_step
      else:
        break
    
    plt.plot(t_list, response)
    plt.grid(True)
    plt.xlabel("Time (seconds)")
    plt.ylabel("MFC Response (LPM)")
    plt.title("Unit Step Response of MFC Simulation")
    plt.draw()  # draw() is non-blocking so test can continue
    
    # Test the MFC after running
    self.assertAlmostEqual(test_mfc1.get_state(), K, places=3,
                           msg="Final value of MFC1 not close to expected value")
    self.assertEqual(test_mfc1.get_time(), time,
                     "MFC1 simulation time out of sync with global simulation time")
    
    # Test the second MFC in simulation
    self.assertTrue(test_mfc2.get_state() != y0_2[0],
                    "MFC state failed to update from y0")
    self.assertAlmostEqual(test_mfc2.get_state(), K, places=3,
                           msg="Final value of MFC2 not close to expected value")
    self.assertEqual(test_mfc2.get_time(), time,
                     "MFC2 simulation time out of sync with global simulation time")
    
    plt.show()  # called at end to prevent plot from automatically closing

  def test_controller_class(self):
    """
    Tests the controller class, ensuring correct initialization and that its
    methods work.
    """
    
    # Initialize constants and lists used for test
    K_list = [10.0, 5.0]
    tau_list = [2.0, 1.0]
    y0 = [0]
    mfc_list = []
    t_step_ctrl = 0.1  # control update rate (seconds)
    Kp_list = [1.5, 2.0]
    control_law_list = []
    mass_flow_des = [3.0, 2.0]  # desired flow rates per MFC
    time = 0.0
    t_step = 0.01
    t_list = []
    stop_time = 10.0
    response = []
    
    # Initialize list of MFCs
    for K, tau in K_list, tau_list:
      mfc_list.append(lab_hardware.MFC(lambda t, y, u: test_ode(t, y, u, K, tau),
                                       lambda x: x[0], y0))
    
    # Initialize list of control laws
    for K in Kp_list:
      control_law_list.append(lambda y, ref: test_ctrl_law(y, ref, K))
    
    # Initialize the controller object
    test_ctrl = lab_hardware.Controller(mfc_list, control_law_list, t_step_ctrl)
    
    # Test initialization
    self.assertIsInstance(test_ctrl, lab_hardware.Controller,
                          "Controller not initialized to correct class.")
    
    # Run the controller
    while time < stop_time:
      if test_ctrl.update(mass_flow_des, t_step, time):
        t_list.append(time)
        response.append(test_ctrl.get_state())
        if time == 0.0 or time % 1.0 < t_step:
          print("{}\t{}".format(time, test_ctrl.get_state()))
        time += t_step
      else:
        break
    
    plt.plot(t_list, response)
    plt.grid(True)
    plt.xlabel("Time (seconds)")
    plt.ylabel("MFC Responses (LPM)")
    plt.title("Controlled Response of MFCs")
    plt.draw()  # draw() is non-blocking so test can continue
    
    #TODO test the result of running the controller
    
    plt.show()  # called at end to prevent plot from automatically closing
    

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testLabHardware']
  unittest.main()