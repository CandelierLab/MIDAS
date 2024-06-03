'''
Agents
'''

import random
import numpy as np

# === Geometry =============================================================

class foop:
  '''
  Field of orientated points
  '''
  def __init__(self, X, Y, A, boxSize, periodic_boundary_condition=True):
    self.X = X
    self.Y = Y
    self.A = A
    self.boxSize = boxSize
    self.periodic_boundary_condition = periodic_boundary_condition

  def center(self, tx, ty, ta=0):
    '''
    (tx,ty) is the target position, which will be at (0,0) after translation
    ta is the target angle, which will be at 0 after rotation
    '''

    # Translation
    if self.periodic_boundary_condition:
      X = (self.X - tx + self.boxSize/2) % self.boxSize - self.boxSize/2
      Y = (self.Y - ty + self.boxSize/2) % self.boxSize - self.boxSize/2
    else:
      X = self.X - tx
      Y = self.Y - ty

    # Rotation
    if ta!=0:
      Z = X+Y*1j
      Z = np.abs(Z)*np.exp(1j*(np.angle(Z)-ta))
      X = np.real(Z)
      Y = np.imag(Z)

    return foop(X, Y, self.A-ta, self.boxSize, self.periodic_boundary_condition)

  def near(self, r, include_self=False):
    '''
    List of points within a given radius around the origin
    '''
    
    # Find nearest
    if include_self:
      I = np.argwhere(np.abs(self.X + 1j*self.Y)<=r).flatten()
    else:
      I = np.argwhere((np.abs(self.X + 1j*self.Y)>0) & (np.abs(self.X + 1j*self.Y)<=r)).flatten()

    return I

  def save(self, file, step):
    '''
    Save data points
    '''

    file.pos[step,:,0] = self.X
    file.pos[step,:,1] = self.Y
    file.pos[step,:,2] = self.A

# === Generic mobile agent ===================================================

class agent:
  '''
  Generic mobile agent (parent class)  
  '''

  def __init__(self, **kwargs):

    # --- Definitions

    # Simulation engine
    self.engine = kwargs['engine']

    # Agent index
    self.idx = kwargs['idx']

    # Group index
    self.gidx = kwargs['gidx']

    # Group name
    self.name = kwargs['name']

    # Orientational noise
    self.noise = kwargs['noise']

    # Kinematic properties
    self.v = None

    self.perceived = None
    self.rho = None
    self.theta = None
    self.alpha = None

    # Density
    self.density = None
    self.kde_sigma = None

    # Trace
    self.trace = None

  def __str__(self):

    if self.__class__.__name__ == 'agent':
      s = '--- agent ---'
    else:
      s = '--- ' + str(self.__class__.__name__) + ' agent ---'

    for key,val in self.__dict__.items():
      s+= '\n' + key + ': ' + str(val)

    return s

  def add_information(self):
    '''
    Display parameters in animation information.
    
    This method is meant to be overloaded.
    '''
    pass

  def setInitialCondition(self, IC):

    # --- Position

    if IC['position'] is None:
      self.x = random.random()*self.engine.boxSize
      self.y = random.random()*self.engine.boxSize      
    else:
      self.x = IC['position'][0]
      self.y = IC['position'][1]
     
    # --- Orientation

    if IC['orientation'] is None:
      self.a = random.random()*2*np.pi
    else:
       self.a = IC['orientation']

    # --- Speed

    self.v = IC['speed']

  def update(self, iteration, F, id=None):
    '''
    To be overloaded
    '''
    pass

  def get_color(self, **kwargs):
    '''
    To be overloaded.
    Defines the agent's color when dynamic_map='custom'.
    '''
    return 0

  def perceive(self, F, r=None, reorient=True, include_self=False):
    '''
    Updating perception of the surroundings

    - Sets density field in polar coordinates around the agent
    - Computes the local density
    '''

    # Center around agent
    if reorient:
      C = F.center(self.x, self.y, self.a)
    else:
      C = F.center(self.x, self.y)

    # Polar coordinates
    Z = C.X + C.Y*1j

    # --- Density

    # Find neighbors
    if self.engine.periodic_boundary_condition or r is not None:
      
      self.perceived = np.array(C.near(0.5 if r is None else r, include_self=include_self), dtype=np.uint16)
      
    else:
      self.perceived = np.arange(self.engine.agents.N)
      if not include_self:
        self.perceived = np.setdiff1d(self.perceived, self.idx)

    # Density
    self.density = np.sum(np.exp(-(np.abs(Z)/self.kde_sigma)**2/2))

    # --- Polar coordinates

    self.rho = np.abs(Z[self.perceived])
    self.theta = np.mod(np.angle(Z[self.perceived]), 2*np.pi)
    self.alpha = np.mod(C.A[self.perceived], 2*np.pi)
    # self.alpha = np.mod(C.A[self.perceived]+np.pi, 2*np.pi) - np.pi

  def move(self):
    '''
    Move the agent
    Bounday conditions can be 'periodic' or 'bouncing'
    '''

    # Angular noise
    if isinstance(self.noise, dict):
      self.a += self.noise['angular']*np.random.randn(1)
      v = self.v + self.noise['velocity']*np.random.randn(1)
    else:
      self.a += self.noise*np.random.randn(1)
      v = self.v

    # Position
    if self.engine.periodic_boundary_condition:
        
      self.x = ((self.x + v*np.cos(self.a)) % self.engine.boxSize)[0]
      self.y = ((self.y + v*np.sin(self.a)) % self.engine.boxSize)[0]

    else:

      x = self.x + v*np.cos(self.a)
      y = self.y + v*np.sin(self.a)

      if x<0:
        x = -x
        self.a = np.pi-self.a
      elif x>self.engine.boxSize:
        x = 2*self.engine.boxSize-x
        self.a = np.pi-self.a

      if y<0:
        y = -y
        self.a = -self.a
      elif y>self.engine.boxSize:
        y = 2*self.engine.boxSize-y
        self.a = -self.a

      self.x = x[0]
      self.y = y[0]

