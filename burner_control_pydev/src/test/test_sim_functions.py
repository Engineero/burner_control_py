'''
Created on Jan 22, 2016

@author: nathantoner
'''

import unittest
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
    """Tests for first_order_delay function."""
    
    pass
  
  def test_first_order_output(self):
    """Tests for first_order_output function."""
    
    pass
  
  def test_static_model(self):
    """Tests for static_model function."""
    
    pass


if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()