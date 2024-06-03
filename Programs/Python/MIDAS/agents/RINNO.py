'''
RINNO agents

Radial Input, Neural Network Outcome
'''

import project
from agents.agent import *

from AE.Display.Animation.Items_2d import *
from animation import piechart

# === Weights ==============================================================

class RINNO_scope(object):

  def __init__(self, nSlices):
    
    self.nSlices = nSlices

    self._foreground = np.zeros(self.nSlices)
    self._background = np.zeros(self.nSlices)

    # Display
    self.display = False

  @property
  def foreground(self):
    return self._foreground

  @foreground.setter
  def foreground(self, v):

    # Convert scalar to numpy arrays
    if isinstance(v, (float, int)):
      v *= np.ones(self.nSlices)

    # Convert list to numpy arrays
    if isinstance(v, list):
      v = np.array(v)

    # Symmetrize if half the weights are provided, otherwise check size
    if v.size==self.nSlices/2:      
      v = np.concatenate((v, np.flip(v)))
    elif v.size!=self.nSlices:
      raise Exception('The number of weights shoudl equal the number of slices.')

    self._foreground = v

  @property
  def background(self):
    return self._background

  @background.setter
  def background(self, v):

    # Convert scalar to numpy arrays
    if isinstance(v, (float, int)):
      v *= np.ones(self.nSlices)

    # Convert list to numpy arrays
    if isinstance(v, list):
      v = np.array(v)

    # Symmetrize if half the weights are provided, otherwise check size
    if v.size==self.nSlices/2:
      v = np.concatenate((v, np.flip(v)))
    elif v.size!=self.nSlices:
      raise Exception('The number of weights shoudl equal the number of slices.')

    self._background = v

  def isdisplayed(self):
    return self.display or (np.count_nonzero(self.foreground) + np.count_nonzero(self.background) > 0)

  def to_list(self):
    return [self.foreground, self.background]

class RINNO_inputs():

  def __init__(self, nSlices, nGroups):
    
    # Axes
    self.FB = [RINNO_scope(nSlices) for i in range(nGroups)]
    self.LR = [RINNO_scope(nSlices) for i in range(nGroups)]
    self.UD = [RINNO_scope(nSlices) for i in range(nGroups)]
    
class RINNO_weights():
  '''
  Weights structure:
    self.input.axis[group].scope[slice] (scope being forground/background)
  For instance:
    self.presence.FB[0].foreground[2]
  '''

  def __init__(self, nSlices, **kwargs):

    # --- Initializations

    # Number of slices
    self.nSlices = nSlices

    # Number of groups
    self.nGroups = kwargs['nGroup'] if 'nGroup' in kwargs else 1

    # --- Inputs

    self.presence = RINNO_inputs(self.nSlices, self.nGroups) if 'presence' in kwargs else None
    self.orientation = RINNO_inputs(self.nSlices, self.nGroups) if 'orientation' in kwargs else None

# === Thresholded Angular Perceptive Agents ================================

class RINNO_2d(agent):
  '''
  2d RINNO 
  '''

  def __init__(self, threshold=0.1, weights=RINNO_weights(4),
               v_min = None, v_max = None, da_max=np.pi/2, dv_max=0.01, delta=0, rmax=None,
               **kwargs):

    # Generic agent constructor
    super().__init__(**kwargs)

    # --- Definitions

    self.threshold = threshold
    self.weights = weights
    self.v_min = v_min
    self.v_max = v_max
    self.da_max = da_max
    self.dv_max = dv_max
    self.delta = delta
    self.rmax = rmax

    # Number of groups, slices
    self.ng = self.weights.nGroups
    self.ns = self.weights.nSlices

    # Initial speed
    # NB: defined later, with initial conditions.
    self.v0 = None

  def information(self):
    '''
    Agent information
    '''

    s = ''

    # Threshold
    s += '<i>r<sub>th</sub></i> = {:.03f}<br>'.format(self.threshold)
    
    # Velocity
    s += '<i>&delta;v</i><sub>max</sub> = {:.03f}<br>'.format(self.dv_max)
    s += 'v<sub>0</sub> = {:.03f}<br>'.format(self.v0)
    if self.v_min is not None:
      s += 'v<sub>min</sub> = {:.03f}<br>'.format(self.v_min)
    if self.v_max is not None:
      s += 'v<sub>max</sub> = {:.03f}<br>'.format(self.v_max)

    # Maximal reorientation
    s += '<i>&delta;&alpha;</i><sub>max</sub> = {:.03f}<br>'.format(self.da_max)

    # Orientational bias
    s += '<i>&delta;</i> = {:.03f}<br>'.format(self.delta)

    # Noise
    s += '<i>&sigma;</i><sub>noise</sub> = {:.03f}<br>'.format(self.noise)

    return s

  def add_info_weights(self):

    # --- Presence weights
      
    if self.weights.presence is not None and self.weights.presence.FB[0].isdisplayed():

      # Description
      self.engine.window.information.add(text, 'FB_presence_text',
        stack = True,
        string = 'F/B presence weights',
        color = 'white',
        fontsize = 10,
      )
      
      # Pie chart
      self.engine.window.information.add(piechart, 'FB_presence_weights',
        stack = True,
        values = self.weights.presence.FB[0].to_list(),
        delta = self.delta,
        radius = 0.07,
        fontsize = 6,
        ratio = 0.6,
        cmap = self.engine.animation.pie_cmap,
      )

    if self.weights.presence is not None and self.weights.presence.LR[0].isdisplayed():

      # Description
      self.engine.window.information.add(text, 'LR_presence_text',
        stack = True,
        string = 'L/R presence weights',
        color = 'white',
        fontsize = 10,
      )
      
      # Pie chart
      self.engine.window.information.add(piechart, 'LR_presence_weights',
        stack = True,
        values = self.weights.presence.LR[0].to_list(),
        delta = self.delta,
        radius = 0.07,
        fontsize = 6,
        ratio = 0.6,
        cmap = self.engine.animation.pie_cmap,
      )

    # --- Orientation weights

    if self.weights.orientation is not None and self.weights.orientation.LR[0].isdisplayed():

      # Description
      self.engine.window.information.add(text, 'LR_orientation_text',
        stack = True,
        string = 'L/R orientation weights',
        color = 'white',
        fontsize = 10,
      )
      
      # Pie chart
      self.engine.window.information.add(piechart, 'LR_orientation_weights',
        stack = True,
        values = self.weights.orientation.LR[0].to_list(),
        delta = self.delta,
        radius = 0.07,
        fontsize = 6,
        ratio = 0.6,
        cmap = self.engine.animation.pie_cmap,
      )

  def update_info_weights(self):

    # --- Presence weights

    if self.weights.presence.LR[0].isdisplayed():

      pie = self.engine.window.information.composite['LR_presence_weights']
      items = self.engine.window.information.item
      
      for i in range(self.ns):

        # Foreground
        items[f'LR_presence_weights_pie_{i:d}_foreground'].colors = (pie.val2color(self.weights.presence.LR[0].foreground[i]), None)

        # Background
        items[f'LR_presence_weights_pie_{i:d}_background'].colors = (pie.val2color(self.weights.presence.LR[0].background[i]), None)

  def setInitialCondition(self, IC):

    self.v0 = IC['speed']

    if self.v_max is None:
      self.v_max = np.max(self.v0) 

    # Parent method
    super().setInitialCondition(IC)

  def update(self, iteration, F, **kwargs):
    '''
    Update angles and move
    '''

    # Update perception
    self.perceive(F, r=self.rmax, include_self=False)

    # Orientational shift
    theta = np.mod(self.theta + self.delta, 2*np.pi)
    
    # --- Inputs -----------------------------------------------------------

    # Indices
    IB = list(range(int(self.ns/4), int(3*self.ns/4)))
    IF = [x for x in list(range(self.ns)) if x not in set(IB)]
    IL = list(range(0, int(self.ns/2)))
    IR = list(range(int(self.ns/2), self.ns))

    presence = np.zeros((self.ng, self.ns, 2))
    orientation = np.zeros((self.ng, self.ns, 2))
    
    mean_orientation = lambda x : np.angle(np.exp(1j*x).sum())

    for i, name in enumerate(self.engine.agents.groupnames):
        
      I = np.isin(self.perceived, self.engine.agents.groups[name])

      for k in range(self.ns):

        # Indices
        K0 = I & (theta>=2*k*np.pi/self.ns) & (theta<=2*(k+1)*np.pi/self.ns) & (self.rho<=self.threshold)
        K1 = I & (theta>=2*k*np.pi/self.ns) & (theta<=2*(k+1)*np.pi/self.ns) & (self.rho>self.threshold)

        # Presence
        presence[i,k,0] = np.count_nonzero(K0)
        presence[i,k,1] = np.count_nonzero(K1)

        # Orientation
        sign = (k in IL) - (k in IR)
        orientation[i,k,0] = sign*mean_orientation(self.alpha[K0])
        orientation[i,k,1] = sign*mean_orientation(self.alpha[K1])

    # --- Updates ----------------------------------------------------------

    # Perceptron inputs
    vF = 0
    vB = 0
    vL = 0
    vR = 0

    # --- Presence FB

    if self.weights.presence is not None:

      # Weights
      wF = np.zeros((self.ng, self.ns, 2))
      wB = np.zeros((self.ng, self.ns, 2))

      for i in range(self.ng):
        wF[i, IF,0] = self.weights.presence.FB[i].foreground[IF]
        wF[i, IF,1] = self.weights.presence.FB[i].background[IF]
        
        wB[i, IB,0] = -self.weights.presence.FB[i].foreground[IB]
        wB[i, IB,1] = -self.weights.presence.FB[i].background[IB]

      # Update FB
      vF += np.sum(np.multiply(presence, wF))
      vB += np.sum(np.multiply(presence, wB))

    # --- Presence LR

    if self.weights.presence is not None:

      # Weights
      wL = np.zeros((self.ng, self.ns, 2))
      wR = np.zeros((self.ng, self.ns, 2))

      for i in range(self.ng):
        wL[i, IL,0] = self.weights.presence.LR[i].foreground[IL]
        wL[i, IL,1] = self.weights.presence.LR[i].background[IL]
  
        wR[i, IR,0] = self.weights.presence.LR[i].foreground[IR]
        wR[i, IR,1] = self.weights.presence.LR[i].background[IR]

      # Update LR
      vL += np.sum(np.multiply(presence, wL))
      vR += np.sum(np.multiply(presence, wR))

    # --- Orientation LR

    if self.weights.orientation is not None:

      # Weights
      wL = np.zeros((self.ng, self.ns, 2))
      wR = np.zeros((self.ng, self.ns, 2))

      for i in range(self.ng):
        wL[i,IL,0] = self.weights.orientation.LR[i].foreground[IL]
        wL[i,IL,1] = self.weights.orientation.LR[i].background[IL]
  
        wR[i,IR,0] = self.weights.orientation.LR[i].foreground[IR]
        wR[i,IR,1] = self.weights.orientation.LR[i].background[IR]

      # Update LR
      vL += np.sum(np.multiply(orientation, wL))
      vR += np.sum(np.multiply(orientation, wR))

    # --- Reorientation

    self.a += self.da_max*(4/np.pi*np.arctan(np.exp((vL-vR)/2))-1)

    # --- Speed

    self.v = self.v_max*2/np.pi*np.arctan(np.exp((vF-vB)/2))

    # self.v = self.v_max*min(iteration/50, 1)

    # NB: Increments cause freezing
    # self.v += self.dv_max*(4/np.pi*np.arctan(np.exp((vF-vB)/2))-1)
    # if self.v_min is not None and self.v<self.v_min: self.v = self.v_min
    # if self.v_max is not None and self.v>self.v_max: self.v = self.v_max

    # Move
    self.move()