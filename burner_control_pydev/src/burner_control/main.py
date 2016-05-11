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
  K = 0.1  # static pressure sensor gain, V/psi (?)
  offset = 1.0  # static pressure sensor offset, V (?)
  K_mfcs = [99.21, 7.968, 7.964, 8.007]  # MFC gains: [air, pilot, middle, outer], SLPM/V
#   tau_mfcs = [1.5, 2.5, 3.0, 1.4]  # MFC time constants, sec
  zeta_mfcs = [0.7054, 0.6, 0.6, 0.6]  # MFC damping ratios
  wn_mfcs = [1.3501, 1, 1, 1]  # MFC natural frequencies, rad/sec
  td_mfcs = [1.8, 0.139, 0.194, 0.306]  # MFC time delays, sec
  y0 = [0.0, 0.0, 0.0, 0.0]  # initial value, 2nd order sys with 3,2 order Pade approximation of time delay
  mfc_list = []
  control_law_list = []
  Q = np.ndarray([[100, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0] [0, 0, 0, 1]])
  R = np.ndarray([[1]])
  sensor_locations = [43.81, 76.2, 236.728, 609.6]  # sensor locations from base, mm
    
  # Initialize sensor list
  sensor_list = [lab_hardware.StaticSensor(model=lambda y: sim_functions.static_model(y, K, offset),
                              location=0.0)]
  for loc in sensor_locations:
    sensor_list.append(lab_hardware.DynamicSensor(model=tf1, location=loc))
  #TODO figure out sensor transfer functions or models!
  
  # Initialize mass flow controller (MFC) list and control law list
  for K, zeta, wn, td in zip(K_mfcs, zeta_mfcs, wn_mfcs, td_mfcs):
    A, B, C = sim_functions.get_second_ord_matrices(K, zeta, wn, td)
    K_lqr = sim_functions.make_lqr_law(A, B, Q, R)
    mfc_list.append(lab_hardware.MFC(lambda t, y, u: sim_functions.system_state_update(t, y, u, A, B),
                                     lambda y: sim_functions.system_output(y, C),
                                     y0))
    control_law_list.append(lambda e: -K_lqr.dot(e))
    
  combustor = lab_hardware.Combustor(p_atm, t_step, t_step_ctrl, sensor_list,
                                     mfc_list, control_law_list)
  

if __name__ == '__main__':
  run_simulation()