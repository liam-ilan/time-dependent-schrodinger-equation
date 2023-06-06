import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.constants as constants

def render(path, thumb_path, config):
  # time
  start = time.time()

  # constants
  hbar = constants.hbar
  m = config['m']

  # initial wave packet conditions
  k = config['k']
  std_dev = config['std_dev']
  x_0 = config['x_0']

  # initialization conditions for field
  bounds = config['bounds']
  sample_number = config['sample_number']

  # potential
  pot_func = config['pot_func']
  V = [pot_func(x) for x in np.linspace(bounds[0], bounds[1], sample_number)]

  # delta x and delta t for finite difference method
  delta_x = abs(bounds[1] - bounds[0]) / (sample_number - 1)
  delta_t = config['delta_t']

  # animation / simulation settings
  fps = config['fps']
  anim_duration = config['anim_duration']
  real_time = config['real_time']

  # number of animation frames / number of simulation steps
  anim_n = fps * anim_duration
  sim_n = int(real_time / delta_t)

  # ratio of simulated to rendered frames
  render_ratio = int(sim_n / anim_n)

  # normalize such that the wave function amplitude squared is a valid PDF
  def normalize(x):
    I = np.sum(np.abs(x)**2 * delta_x)
    res = x / (I**0.5)
    return res


  # generate initial wave packet
  def gaussian_wave_packet(x):
    return normalize(np.exp(1j * k * x) * np.exp(-0.5 * ((x - x_0) / std_dev)**2))


  # quantum field
  # each item is a different point in space
  field = gaussian_wave_packet(np.linspace(bounds[0], bounds[1], sample_number))

  # list of fields to be animated
  animation_frames = []

  # gets field after a time delta_t, given inital field
  def get_next_field(x):

    # bounds are set to 0
    res = [0]

    # skip 1st and last items, as they are boundaries
    for n in range(1, sample_number - 1):
      term_1 = ((1j) / (2 * m)) * ((delta_t) / (delta_x**2)) * (x[n + 1] - 2 * x[n] + x[n - 1])
      term_2 = -(1j / hbar) * delta_t * V[n] * x[n]
      term_3 = x[n]
      
      res.append(term_1 + term_2 + term_3)

    res.append(0)

    return normalize(res)


  # simulate future timesteps and appen to field
  print(f'Simulating {sim_n} Timesteps')
  for i in range(sim_n - 1):
    if i % render_ratio == 0:
      animation_frames.append(np.copy(field))

    field = get_next_field(field)

  print('Finished Simulation')

  # visualization
  print("Started Visualization")
  plt.figure(figsize=(16,9))

  # set up axes
  prob_axis = plt.gca()
  comp_axis = prob_axis.twinx()
  pot_axis = prob_axis.twinx()

  # scientific notation for all numbers
  prob_axis.ticklabel_format(scilimits = (0, 0))
  comp_axis.ticklabel_format(scilimits = (0, 0))
  pot_axis.ticklabel_format(scilimits = (0, 0))

  # set Re(psi) and Im(psi) axis to be on left
  comp_axis.spines.right.set_position(("axes", 0))

  # animation function called on every animation frame
  def animate(i):
    pdf.set_data(np.linspace(bounds[0], bounds[1], sample_number),
                    np.abs(animation_frames[i]) ** 2)
    real.set_data(np.linspace(bounds[0], bounds[1], sample_number),
                    np.real(animation_frames[i]))
    imag.set_data(np.linspace(bounds[0], bounds[1], sample_number),
                    np.imag(animation_frames[i]))

    time_elapsed = i * real_time / len(animation_frames)

    time_str = fr'{round(time_elapsed / 10 ** get_sci(time_elapsed),2):0<4} \times 10^{{{get_sci(time_elapsed)}}}'
    if time_elapsed == 0: time_str = 0
    legend.get_title().set_text(fr'$t = {time_str} s$')
    return [pot, pdf, real, imag]

  # initial plotting
  print('Plotting Initial Conditions')
  pdf = prob_axis.plot(np.linspace(bounds[0], bounds[1], sample_number),
                  np.abs(animation_frames[0]) ** 2, color='tab:blue')[0]
  real = comp_axis.plot(np.linspace(bounds[0], bounds[1], sample_number),
                  np.real(animation_frames[0]), color='tab:orange', linestyle='dotted')[0]
  imag = comp_axis.plot(np.linspace(bounds[0], bounds[1], sample_number),
                  np.imag(animation_frames[0]), color='tab:green', linestyle='dotted')[0]
  pot = pot_axis.plot(np.linspace(bounds[0], bounds[1], sample_number),
                  V, color='tab:red')[0]

  # set axes for real and imag to be symetric
  comp_axis.set_ylim([-max(comp_axis.get_ylim()), max(comp_axis.get_ylim())])

  # give prob and comp axes 25 percent buffer
  comp_axis.set_ylim([i * 1.25 for i in comp_axis.get_ylim()])
  prob_axis.set_ylim([i * 1.25 for i in prob_axis.get_ylim()])

  # legend
  legend = plt.legend(
    [pdf, real, imag, pot], 
    ['$|\psi(x, t)|^2$', '$Re(\psi(x, t))$', '$Im(\psi(x, t))$', '$V(x)$'],
    title='$t = 0s$',
    loc='upper right'
  )

  # hide scientific notation text (will be added into label later)
  prob_axis.yaxis.get_offset_text().set_visible(False)
  comp_axis.yaxis.get_offset_text().set_visible(False)
  pot_axis.yaxis.get_offset_text().set_visible(False)

  prob_axis.xaxis.get_offset_text().set_visible(False)

  # returns exponent for scientific notation
  def get_sci(n):

    # special case 0 handled with 1 (though this is improper)
    if n == 0: 
      return 1
    return math.floor(math.log(abs(n), 10))

  # returns exponent for scientific notation given limits
  def get_sci_lim(lim):
    return max(get_sci(lim[0]), get_sci(lim[1]))

  prob_axis.set_ylabel(f'$|\psi(x, t)|^2$ ($10^{{{get_sci_lim(prob_axis.get_ylim())}}}$)')
  comp_axis.set_ylabel(f'$Re(\psi(x, t))$ and $Im(\psi(x, t))$ ($10^{{{get_sci_lim(comp_axis.get_ylim())}}} m^{{-3/2}}$)')
  pot_axis.set_ylabel(f'V(x) ($10^{{{get_sci_lim(pot_axis.get_ylim())}}} J$)')

  prob_axis.set_xlabel(f'x ($10^{{{get_sci_lim(prob_axis.get_xlim())}}} m$)')

  # title
  plt.title(fr'''Time Dependent Schrodinger Equation Evolution for Initial Gaussian Wave Packet
  $k={round(k/10**get_sci(k),2):0<4} \times 10^{{{get_sci(k)}}} m^{{-1}}$, 
  $\sigma={round(std_dev/10**get_sci(std_dev),2):0<4} \times 10^{{{get_sci(std_dev)}}} m$
  $m={round(m/10**get_sci(m), 2):0<4} \times 10^{{{get_sci(m)}}} kg$''')

  print('Saving Initial Conditions')
  plt.savefig(thumb_path)
  
  # animate
  print('Creating Animation')
  anim = animation.FuncAnimation(plt.gcf(),
                                animate,
                                frames=len(animation_frames),
                                blit=True)

  print('Saving Animation')
  FFwriter = animation.FFMpegWriter(fps=fps)
  anim.save(path, writer=FFwriter)

  print('Closing')
  plt.close()

  print(f'Done in {time.time() - start} s')
