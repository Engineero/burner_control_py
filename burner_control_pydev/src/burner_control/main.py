'''
Created on Jan 7, 2016

@author: nathantoner
'''

import numpy as np
from burner_control import lab_hardware, sim_functions

def run_simulation():
  """Run a simulation of the controller."""
  
  # Define system parameters
  p_atm = 14.0  # psi
  t_step = 0.01  # simulation time step, seconds
  t_step_ctrl = 0.1  # controller update rate, seconds
  K = 0.1
  offset = 1.0
  K_mfcs = [10.0, 5.0, 2.0, 1.0]  # MFC gains
  tau_mfcs = [1.5, 2.5, 3.0, 1.4]  # MFC time constants
  td_mfcs = [0.2, 0.1, 0.4, 0.3]  # MFC time delays
  y0 = [0.0, 0.0, 0.0]  # initial value
  mfc_list = []
  control_law_list = []
  Q = np.ndarray([[100, 0, 0], [0, 1, 0], [0, 0, 1]])
  R = np.ndarray([[1]])
    
  # Initialize sensor list
  sensor_list = [lab_hardware.StaticSensor(model=lambda y: sim_functions.static_model(y, K, offset),
                              location=0.0),
                 lab_hardware.DynamicSensor(model=tf1, location=1.0)]
  #TODO initialize others
  #TODO figure out transfer functions!
  
  # Initialize mass flow controller (MFC) list and control law list
  for K, tau, delay in zip(K_mfcs, tau_mfcs, td_mfcs):
    A, B, C = sim_functions.get_state_matrices(K, tau, delay)
    K_lqr = sim_functions.make_lqr_law(A, B, Q, R)
    mfc_list.append(lab_hardware.MFC(lambda t, y, u: sim_functions.first_order_delay(t, y, u, A, B),
                                     lambda y: sim_functions.first_order_output(y, C),
                                     y0))
    control_law_list.append(lambda e: -K_lqr.dot(e))
    
  combustor = lab_hardware.Combustor(p_atm, t_step, t_step_ctrl, sensor_list,
                                     mfc_list, control_law_list)
  

if __name__ == '__main__':
  run_simulation()