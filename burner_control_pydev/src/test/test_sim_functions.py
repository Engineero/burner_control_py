'''
Created on Jan 22, 2016

@author: nathantoner
'''

import unittest
import itertools
from scipy import integrate
import numpy as np
from burner_control import sim_functions


class Test(unittest.TestCase):
  """Unit tests for simulation functions in sim_functions.py."""

  def test_one_sphere(self):
    """Tests for one_sphere function."""
    
    # Define constants
    radii = [1.0, 2.0, 3, 4, 100]
    bad_radii = [0, 0.0, -1, -2, -0]
    origin_list = [0, 0.0, [0, 0], [0, 0.0], [0, 0, 0, 0, 0, 0, 0]]
    default_bad = [1, 1.0, [1, 0], [0, 1], [0, 0, 1], [0.5, 0.5, 0.1, -0.1]]
    default_good = [1.1, 2, -1.1, [2, 0], [-1, 1], [-1, 0.5, 0.5, 0.1]]
    assigned_good = [100.1, [100, 1], [-100, 0.1], [100, -1, 1, 0]]
    
    # Test the default function implementation
    for origin in origin_list:
      self.assertFalse(sim_functions.one_sphere(origin),
                       "Default one_sphere misclassifies origin.")
    for point in default_bad:
      self.assertFalse(sim_functions.one_sphere(point),
                       "Default one_sphere misclassifies points inside sphere.")
    for point in default_good:
      self.assertTrue(sim_functions.one_sphere(point),
                      "Default one_sphere misclassifies points outside sphere.")
    
    # Test ValueError raised when radius = 0
    with self.assertRaises(ValueError):
      for radius in bad_radii:
        sim_functions.one_sphere(0, radius)
    
    # Test that string arguments raise a TypeError
    with self.assertRaises(TypeError):
      sim_functions.one_sphere("0.0")
      sim_functions.one_sphere(["0", "0"])
      sim_functions.one_sphere(0, radius="1")
      sim_functions.one_sphere([0, 0], radius="1")
      sim_functions.one_sphere(0, radius=[1, 1])
      sim_functions.one_sphere([0, 0], radius=[1, 1])
      
    # Test some different radii
    for radius in radii:
      for point in default_bad:
        self.assertFalse(sim_functions.one_sphere(point, radius),
                         "Misclassifying bad points when assigning radius.")
      for point in assigned_good:
        self.assertTrue(sim_functions.one_sphere(point, radius),
                        "Misclassifying good points when assigning radius.")
    
  def test_first_order_delay(self):
    """Tests for system_state_update function for first-order ODE."""
    
    # Define constants
    y0 = np.reshape([0.5]*3, (3, 1))
    t_step = 0.01
    tau_list = [0.1, 0.3, 1]
    delay_list = [0.1, 0.3, 1]
    input_list = [0., 1., -1.]
    param_list = set(itertools.chain(*[zip(x, delay_list) for x in
                                       itertools.permutations(tau_list, len(delay_list))]))
    
    # Set up the ODE
    test_ode = integrate.ode(sim_functions.system_state_update)
    test_ode.set_integrator("dopri5")
    test_ode.set_initial_value(y0)
    
    # Test initialization
    self.assertListEqual(test_ode.y.tolist(), y0.tolist(),
                         "ODE not initialized to correct state.")
    self.assertEqual(test_ode.t, 0, "ODE simulation time not initialized to 0.")
    
    # Test response of the ODE
    print("Testing ODE", end="", flush=True)
    for tau, delay in param_list:
      den = tau*delay**2
      A = np.array([[0, 1, 0], [0, 0, 1], [-12/den, -(6*delay + 12*tau)/den, -(6*tau + delay)/tau/delay]])
      B = np.array([[0], [0], [12/den]])
      for u in input_list:
        test_ode.set_initial_value(y0, 0)
        time = 0
        print(".", end="", flush=True)
        while test_ode.successful() and time < 10*tau+delay:
          # test_ode.set_f_params(*[u, tau, delay])
          test_ode.set_f_params(*[u, A, B])
          test_ode.integrate(test_ode.t + t_step)
          time += t_step
#           if time == 0 or test_ode.t % 1.0 < t_step:
#             print("{}\t{}\t{}".format(test_ode.t, time, test_ode.y))
          
        self.assertEqual(test_ode.t, time,
                         "ODE simulation time does not match global simulation time.")
        self.assertAlmostEqual(test_ode.y.flatten().tolist()[0], u, places=1,
                               msg="ODE final value not equal to expected value.")
        self.assertEqual(test_ode.y.shape, (3, 1), "ODE simulation state not the correct shape")
    print("Done!")
  
  def test_first_order_output(self):
    """Tests for system_output function for first order ODE."""
    
    # Define constants
    y_list = itertools.combinations_with_replacement([-1, 0, 1], 3)
    bad_y_list = [0, 1, -1, [1, 0], [0, 0], [1, 1, 1, 1]]
    K_list = [1, 3, 10]
    delay_list = [0, 0.1, 0.3, 1]
    mean_list = [0, 1, -1]
    std_list = [0.01, 0.03, 0.1, 0.3]
    param_list = set(itertools.chain(*[zip(K_list, x) for x in
                                       itertools.permutations(delay_list, len(K_list))]))
    noise_list = set(itertools.chain(*[zip(mean_list, x) for x in
                                       itertools.permutations(std_list, len(mean_list))]))
    
    # Test response of the output function
    for K, delay in param_list:
      C = np.array([[K, -K*delay/2, K*delay**2/12]])
      for y in y_list:
        self.assertEqual(sim_functions.system_output(np.array(y), C)[0],
                         y[0]*K - y[1]*K*delay/2 + y[2]*K*delay**2/12,
                         "Unexpected output function response for y={}, K={}, delay={}".format(y, K, delay))
        for mean, std in noise_list:  # test response with noise
          self.assertAlmostEqual(sim_functions.system_output(np.array(y), C, mean, std),
                                 y[0]*K - y[1]*K*delay/2 + y[2]*K*delay**2/12 + mean,
                                 delta=4*std,
                                 msg="Output function response not within 4 std of expected.\nNote this may not be an error since ~0.01% of values lie outside of this range.")
    
    # Test that function raises TypeError for inputs of wrong dimension
    with self.assertRaises(TypeError):
      for y in bad_y_list:
        sim_functions.system_output(np.array(y), np.array([[1]]))
    
    # Test that non-ndarray inputs raise AttributeError
    with self.assertRaises(AttributeError):
      for y in bad_y_list:
        sim_functions.system_output(y, np.array([[1]]))
        sim_functions.system_output(np.array(y), 1)
        sim_functions.system_output(np.array(y), [1, 1])
  
  def test_static_model(self):
    """Tests for static_model function."""
    
    # Define constants
    y_list = [0, 1, -1, 0.1, -2.5, 100, 1000, -10000.1]
    K_list = [1, 10, -2]
    offset_list = [0, 1, -2]
    mean = 0
    std_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    param_list = set(itertools.chain(*[zip(x, offset_list) for x in
                                       itertools.permutations(K_list, len(offset_list))]))
    
    # Test the default function implementation
    for y in y_list:
      for K, offset in param_list:
        self.assertEqual(sim_functions.static_model(y, K, offset), K*y+offset,
                         "Default static_model does not return expected value for params y={}, K={}, offset={}".format(y, K, offset))
    
    # Test the function with Gaussian noise
    for y in y_list:
      for K, offset in param_list:
        for std in std_list:
          self.assertNotEqual(sim_functions.static_model(y, K, offset, mean, std),
                                 K*y+offset,
                                 "Static model response not showing noise effect.")
      
    # Test that list and string arguments raise a TypeError
    with self.assertRaises(TypeError):
      sim_functions.static_model([1, 2], 1, 1)
      sim_functions.static_model(1, [1, 2], 2)
      sim_functions.static_model(1, 2, [1, 2])

  def test_make_lqr_law(self):
    """Tests for make_lqr_law function."""
    
    # Define constants
    tau_list = [0.3, 1, 3]
    delay_list = [0.1, 0.3]
    param_list = set(itertools.chain(*[zip(tau_list, x) for x in
                                       itertools.permutations(delay_list, len(tau_list))]))
    Q = np.array([[100, 0, 0], [0, 1, 0], [0, 0, 1]])  # state error weight
    R = np.array([[1]])  # control effort weight
    
    for tau, delay in param_list:
      den = tau*delay**2
      A = np.array([[0, 1, 0], [0, 0, 1], [-12/den, -(6*delay + 12*tau)/den, -(6*tau + delay)/tau/delay]])
      B = np.array([[0], [0], [12/den]])
      
      # Create control law
      K_lqr, _, eig_vals = sim_functions.make_lqr_law(A, B, Q, R)
      print(eig_vals)
    
      # Test return type
      self.assertIsInstance(K_lqr, np.ndarray,
                            "LQR law not returned as ndarray.")
      
      # Test return dimensions
      self.assertTupleEqual(K_lqr.shape, (1, 3),
                       "LQR law not the correct shape: {}, expected (1, 3).".format(K_lqr.shape))
      
      # Test that controlled system is stable
      self.assertTrue(all([np.real(item) < 0 for item in eig_vals]),
                      "All LQR-controlled poles not in left-half plane.")
  
  def test_get_state_matrices(self):
    """Tests the get_state_matrices function."""
    
    # Define constants
    K_list = [1, 3, 10]
    tau_list = [0.1, 0.3, 1]
    delay_list = [0.01, 0.1, 0.3, 1]
    param_list = set(itertools.chain(*[zip(K_list, x, y) for x in
                                       itertools.permutations(tau_list, len(K_list))
                                       for y in
                                       itertools.permutations(delay_list, len(K_list))]))
    for K, tau, td in param_list:
      den = tau*td**2
      A_exp = np.array([[0, 1, 0], [0, 0, 1], [-12/den, -(6*td + 12*tau)/den, -(6*tau + td)/tau/td]])
      B_exp = np.array([[0], [0], [12/den]])
      C_exp = np.array([[K, -K*td/2, K*td**2/12]])
      A, B, C = sim_functions.get_state_matrices(K, tau, td)
      self.assertListEqual(A.tolist(), A_exp.tolist(),
                           "A matrix does not match expected.")
      self.assertListEqual(B.tolist(), B_exp.tolist(),
                           "B matrix does not match expected.")
      self.assertListEqual(C.tolist(), C_exp.tolist(),
                           "C matrix does not match expected")
    
    def test_get_second_ord_matrices(self):
      """Tests the sim_functions.get_second_ord_matrices method."""
      pass
  

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()