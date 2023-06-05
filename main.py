import time
import numpy as np
import scipy.constants as constants
from renderer import render

start = time.time()

# all configuration in SI units
config_free = {
  # wave packet config
  'k': 6e9,
  'std_dev': 9e-10,
  'x_0': -1.5e-8,
  'm': constants.m_e,

  # bounds of simulation
  'bounds': [-2e-8, 2e-8],

  # number of space steps
  'sample_number': 800,

  # time step for simulation
  'delta_t': 5e-54,

  # simulation time
  'real_time': 3e-48,

  # animation settings
  'fps': 30,
  'anim_duration': 15,

  # potential function
  'pot_func': lambda f: 0,
}

config_tunnel = config_free.copy()
config_tunnel['pot_func'] = lambda x: 2e15 if 0 < x < 2.5e-9 else 0
config_tunnel['x_0'] = -1e-8
config_tunnel['real_time'] = 4e-48

config_double = config_free.copy()
config_double['pot_func'] = lambda x: 2e15 if 0 < x < 1.25e-9 or 3.75e-9 < x < 5e-9 else 0
config_double['x_0'] = -1e-8
config_double['real_time'] = 4e-48

config_barrier = config_free.copy()
config_barrier['pot_func'] = lambda x: 1e17 if 0 < x < 2.5e-9 else 0
config_barrier['x_0'] = -1e-8
config_barrier['real_time'] = 4e-48

config_oscillator = config_free.copy()
config_oscillator['pot_func'] = lambda x: 1e17 * ((1 / 1.5e-8) * x) ** 2
config_oscillator['x_0'] = 0
config_oscillator['real_time'] = 4e-48
config_oscillator['bounds'] = [-7.5e-9, 7.5e-9]
config_oscillator['sample_number'] = 300
config_oscillator['std_dev'] = 6e-10

config_quartic = config_free.copy()
config_quartic['pot_func'] = lambda x: 1e16 * (((1 / 5e-9) * x) ** 4 - 2 * ((1 / 5e-9) * x) ** 2)
config_quartic['x_0'] = 0
config_quartic['k'] = 0
config_quartic['real_time'] = 4e-48
config_quartic['bounds'] = [-7.5e-9, 7.5e-9]
config_quartic['sample_number'] = 300
config_quartic['std_dev'] = 6e-10

config_stationary = config_free.copy()
config_stationary['x_0'] = 0
config_stationary['k'] = 0

render('./public/animations/free.mp4', config_free)
render('./public/animations/tunnel.mp4', config_tunnel)
render('./public/animations/double.mp4', config_double)
render('./public/animations/barrier.mp4', config_barrier)
render('./public/animations/oscillator.mp4', config_oscillator)
render('./public/animations/quartic.mp4', config_quartic)
render('./public/animations/stationary.mp4', config_stationary)

print(f'Finished full render job in {time.time() - start} s')