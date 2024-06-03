'''
CR agents
'''

import project
from agents.agent import *

from AE.Display.Animation.Items_2d import *
from animation import piechart

class CR_2d(agent):
  '''
  2d CR
  '''

  def __init__(self, ns, r_light, w_light, th_light=0,  r_dens=None, w_dens=None,
               v_min = None, v_max = None, da_max=np.pi/2, dv_max=0.01, rmax=None,
               **kwargs):

    # Generic agent constructor
    super().__init__(**kwargs)

    # --- Definitions

    self.r_light = r_light
    self.w_light = w_light
    self.th_light = th_light

    self.r_dens = r_dens
    self.w_dens = w_dens

    self.v_min = v_min
    self.v_max = v_max
    self.da_max = da_max
    self.dv_max = dv_max
    self.rmax = rmax

    # Number of groups, slices
    self.ng = 1
    self.ns = ns

    # Initial speed
    # NB: defined later, with initial conditions.
    self.v0 = None

    # --- Light

    self.light_nbins = 1000
    self.light_angles = np.linspace(0, np.pi, self.light_nbins)
    self.light_profile = np.cos(self.light_angles-np.pi/2)

    # Total intensity received (for coloring)
    self.light = 0.5

  def information(self):
    '''
    Agent information
    '''

    s = ''

    # Velocity
    s += '<i>&delta;v</i><sub>max</sub> = {:.03f}<br>'.format(self.dv_max)
    s += 'v<sub>0</sub> = {:.03f}<br>'.format(self.v0)
    if self.v_min is not None:
      s += 'v<sub>min</sub> = {:.03f}<br>'.format(self.v_min)
    if self.v_max is not None:
      s += 'v<sub>max</sub> = {:.03f}<br>'.format(self.v_max)

    # Maximal reorientation
    s += '<i>&delta;&alpha;</i><sub>max</sub> = {:.03f}<br>'.format(self.da_max)

    # Noise
    s += '<i>&sigma;</i><sub>a_noise</sub> = {:.03f}<br>'.format(self.noise['angular'])
    s += '<i>&sigma;</i><sub>v_noise</sub> = {:.03f}<br>'.format(self.noise['velocity'])

    return s

  def add_info_weights(self):

    pass
    # # --- Presence weights
      
    # if self.w_light.presence is not None and self.w_light.presence.FB[0].isdisplayed():

    #   # Description
    #   self.engine.window.information.add(text, 'FB_presence_text',
    #     stack = True,
    #     string = 'F/B presence weights',
    #     color = 'white',
    #     fontsize = 10,
    #   )
      
    #   # Pie chart
    #   self.engine.window.information.add(piechart, 'FB_presence_weights',
    #     stack = True,
    #     values = self.w_light.presence.FB[0].to_list(),
    #     delta = self.delta,
    #     radius = 0.07,
    #     fontsize = 6,
    #     ratio = 0.6,
    #     cmap = self.engine.animation.pie_cmap,
    #   )

    # if self.w_light.presence is not None and self.w_light.presence.LR[0].isdisplayed():

    #   # Description
    #   self.engine.window.information.add(text, 'LR_presence_text',
    #     stack = True,
    #     string = 'L/R presence weights',
    #     color = 'white',
    #     fontsize = 10,
    #   )
      
    #   # Pie chart
    #   self.engine.window.information.add(piechart, 'LR_presence_weights',
    #     stack = True,
    #     values = self.w_light.presence.LR[0].to_list(),
    #     delta = self.delta,
    #     radius = 0.07,
    #     fontsize = 6,
    #     ratio = 0.6,
    #     cmap = self.engine.animation.pie_cmap,
    #   )

    # # --- Orientation weights

    # if self.w_light.orientation is not None and self.w_light.orientation.LR[0].isdisplayed():

    #   # Description
    #   self.engine.window.information.add(text, 'LR_orientation_text',
    #     stack = True,
    #     string = 'L/R orientation weights',
    #     color = 'white',
    #     fontsize = 10,
    #   )
      
    #   # Pie chart
    #   self.engine.window.information.add(piechart, 'LR_orientation_weights',
    #     stack = True,
    #     values = self.w_light.orientation.LR[0].to_list(),
    #     delta = self.delta,
    #     radius = 0.07,
    #     fontsize = 6,
    #     ratio = 0.6,
    #     cmap = self.engine.animation.pie_cmap,
    #   )

  def update_info_weights(self):

    pass
    # # --- Presence weights

    # if self.w_light.presence.LR[0].isdisplayed():

    #   pie = self.engine.window.information.composite['LR_presence_weights']
    #   items = self.engine.window.information.item
      
    #   for i in range(self.ns):

    #     # Foreground
    #     items[f'LR_presence_weights_pie_{i:d}_foreground'].colors = (pie.val2color(self.w_light.presence.LR[0].foreground[i]), None)

    #     # Background
    #     items[f'LR_presence_weights_pie_{i:d}_background'].colors = (pie.val2color(self.w_light.presence.LR[0].background[i]), None)

  def setInitialCondition(self, IC):

    self.v0 = IC['speed']

    # Parent method
    super().setInitialCondition(IC)

  def get_color(self, **kwargs):
    '''
    Defines agent's color based on the normalized amount of light received.
    '''

    return self.light

  def update(self, iteration, F, id=None):
    '''
    Update angles and move
    '''

    # Update perception
    self.perceive(F, r=self.rmax, include_self=False)
    
    # --- Inputs -----------------------------------------------------------

    input_dens = np.zeros((2, self.ns))
    input_light = np.zeros(self.ns)

    light = np.ones(self.light_nbins)
    bconv = lambda y : np.round((self.light_nbins-1)*(y/np.pi))
    
    # --- Shadows

    dtheta = np.arctan(self.r_light/self.rho)
    theta_0 = np.mod(self.a + self.theta - dtheta, 2*np.pi)
    theta_1 = np.mod(self.a + self.theta + dtheta, 2*np.pi)
      
    I0 = bconv(theta_0) 
    I0[theta_0>np.pi] = 0

    I1 = bconv(theta_1)
    I1[theta_1>np.pi] = self.light_nbins-1

    for i0, i1 in zip(I0, I1):
      if i1>i0 and i1-i0<self.light_nbins-1:
        light[int(i0):int(i1)] = 0
    
    # --- Light sensing

    # Shadowed light profile
    light_profile = self.light_profile*light

    for k in range(self.ns):

      # Definitions
      theta_0 = np.mod(self.a + 2*k*np.pi/self.ns, 2*np.pi)
      theta_1 = np.mod(self.a + 2*(k+1)*np.pi/self.ns, 2*np.pi)

      i0 = int(bconv(theta_0)) if theta_0<np.pi else 0
      i1 = int(bconv(theta_1)) if theta_1<np.pi else self.light_nbins-1

      # Input
      input_light[k] = np.trapz(light_profile[i0:i1], self.light_angles[i0:i1]) if (i1>i0 and i1-i0<(self.light_nbins-1)) else 0

    # --- Density

    for k in range(self.ns):

      # Indices
      K0 = (self.theta>=2*k*np.pi/self.ns) & (self.theta<=2*(k+1)*np.pi/self.ns) & (self.rho<=self.r_dens)
      # K1 = (self.theta>=2*k*np.pi/self.ns) & (self.theta<=2*(k+1)*np.pi/self.ns) & (self.rho>self.r_dens)

      # Presence
      input_dens[0,k] = np.count_nonzero(K0)
      # input_dens[k,1] = np.count_nonzero(K1)

    # --- Normalization

    # Normalized total amount of light perceived
    self.light = np.sum(input_light)/2

    # Normalized light input
    if np.sum(input_light):
      input_light /= np.sum(input_light)
    
    # Normalized density
    if np.sum(input_dens):
      input_dens /= np.sum(input_dens)

    # --- Updates ----------------------------------------------------------

    # --- Perceptron output

    output = 0

    # Light
    if self.light>=self.th_light:
      output -= np.sum(np.multiply(input_light, self.w_light))

    # Density
    output += np.sum(np.multiply(input_dens, self.w_dens))

    # print('--------------')
    # print(input_dens)
    # print(self.w_dens)

    # Pre-activation noise
    output += np.random.randn(1)*self.noise['bias']

    # Reorientation
    self.a += self.da_max*(4/np.pi*np.arctan(np.exp((output)/2))-1)

    # Variable orientational noise
    # self.a += np.random.randn(1)*(1-self.light)**2

    # Move
    self.move()