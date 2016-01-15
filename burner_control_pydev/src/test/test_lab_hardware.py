'''
Created on Jan 8, 2016

@author: nathantoner
'''
import unittest
import numpy as np
import matplotlib.pyplot as plt
from burner_control import lab_hardware

class TestLabHardware(unittest.TestCase):


  def test_mfc_update(self):
    """
    Tests the MFC class, ensuring that it is initialized correctly, and that
    its methods work.
    """
    
    # Set up the MFC
    K = 10.0
    tau = 0.5
    td = 0.1
    y0 = [0.0]*3
    test_mfc = lab_hardware.MFC(lambda t, y, u: lab_hardware.first_order_delay(t, y, u, K, tau, td),
                                y0)
    
    # Run initialization tests on the MFC.
    self.assertIsInstance(test_mfc, lab_hardware.MFC,
                          "Failure to initialize MFC class.")
    self.assertListEqual(test_mfc.get_state().tolist(), y0,
                         "MFC state failed to initialize to y0")
    self.assertEqual(test_mfc.get_time(), 0.0,
                     "MFC time failed to initialize to 0.0")
    
    # Run the MFC a bit
    time = 0.0
    t_list = []
    t_step = 0.01
    stop_time = 10.0
    input_val = 1.0
    response = []
    while test_mfc.get_time() < stop_time:
      if test_mfc.update(input_val, t_step):
        time += t_step
        t_list.append(time)
        response.append(test_mfc.get_state()[0])
        if test_mfc.get_time() % 1.0 < t_step: print(test_mfc.get_state())
      else:
        break
    
    plt.plot(t_list, response)
    plt.xlabel("Time (seconds)")
    plt.ylabel("MFC Response (LPM)")
    plt.title("Unit Step Response of MFC Simulation")
    plt.draw()  # draw() is non-blocking so test can continue
    
    # Test the MFC after running
    self.assertTrue(all([a != b for a, b in zip(test_mfc.get_state().tolist(), y0)]),
                    "MFC state failed to update from y0")
    self.assertEqual(test_mfc.get_time(), time,
                     "MFC simulation time out of sync with global simulation time")
    
    plt.show()  # called at end to prevent plot from automatically closing

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testLabHardware']
  unittest.main()