'''
MIDAS core: initial conditions
'''

import numpy as np
from rich import print
from rich.panel import Panel

import MIDAS

class initial_conditions:

  # ────────────────────────────────────────────────────────────────────────
  def __init__(self, group):

    # Define group and arena
    self.group = group
    self.arena = group.engine.geometry.arena


    # Default values
    self.position = None
    self.velocity = None
    self.orientation = None
    self.speed = 0.01

  # ────────────────────────────────────────────────────────────────────────
  def rich(self, title=True):

    s = ''

    if title:
      s += f'[b i #abcdef]Initial conditions[/]\n'
    
    s += f'position: {self.position}\n'
    if self.velocity is not None:
      s += f'position: {self.velocity}\n'
    else:
      s += f'orientation: {self.orientation}\n'
      s += f'speed: {self.speed}\n'

    return s

  # ────────────────────────────────────────────────────────────────────────
  def display(self):
    
    print(Panel(self.rich(title=False), title='Initial conditions'))

  # ────────────────────────────────────────────────────────────────────────
  def get_positions(self):

    if self.position in [None, 'random', 'shuffle']:

      # ─── Random positions ────────────────────

      match self.arena.type:

        case MIDAS.ARENA.RECTANGULAR:

          x = (np.random.rand(self.group.N) - 1/2)*self.arena.shape[0]
          y = (np.random.rand(self.group.N) - 1/2)*self.arena.shape[1]

        case MIDAS.ARENA.CIRCULAR:

          match self.arena.dimension:

            case 2:
              u1 = np.random.rand(self.group.N)
              u2 = np.random.rand(self.group.N)

              x = np.sqrt(u2)*np.cos(2*np.pi*u1)*self.arena.radius
              y = np.sqrt(u2)*np.sin(2*np.pi*u1)*self.arena.radius
          
            case _:
              ''' To implement '''
              pass

      # Output
      return (x, y)

    elif isinstance(self.position, dict):
      
      # ─── Initial patterns ────────────────────

      match self.position['pattern']:
        
        case 'concentrated':
          ''' Concentration in a circle of given radius.'''

          match self.arena.dimension:

            case 2:
              u1 = np.random.rand(self.group.N)
              u2 = np.random.rand(self.group.N)

              x = np.sqrt(u2)*np.cos(2*np.pi*u1)*self.position['radius']
              y = np.sqrt(u2)*np.sin(2*np.pi*u1)*self.position['radius']

        case 'condensed':
          ''' Condensed in a Gaussian density field of given size.'''

          match self.arena.dimension:

            case 2:
              x = np.random.randn(self.group.N)*self.position['sigma']
              y = np.random.randn(self.group.N)*self.position['sigma']

      # Output
      return (x, y)

    elif isinstance(self.position, (list, np.ndarray)):

      # ─── User-defined positions ──────────────

      pos = np.array(self.position)
      return (pos[:,0], pos[:,1])
    
  # ────────────────────────────────────────────────────────────────────────
  def get_velocities(self):

    if self.velocity in [None, 'random', 'shuffle']:

      # ─── Random velocities ───────────────────

      # ─── Speed

      if self.speed in [None, 'random', 'shuffle']:

        # TODO: check that vmin and vmax are correctly defined

        v = self.group.vmin + (self.group.vmax - self.group.vmin)*np.random.rand(self.group.N)

      elif isinstance(self.speed, (list, np.ndarray)):
        v = np.array(self.speed)

      else:
        v = np.full((self.group.N), fill_value=self.speed)

      # ─── Orientation

      match self.arena.dimension:

        case 2:

          if self.orientation in [None, 'random', 'shuffle']:
            a = 2*np.pi*np.random.rand(self.group.N)

          elif isinstance(self.orientation, (list, np.ndarray)):
            a = np.array(self.orientation)

          # Output
          return (v, a)
        
        case _:

          # TODO
          pass
    
    elif isinstance(self.velocity, (list, np.ndarray)):

      # ─── User-defined velocities ─────────────

      vel = np.array(self.velocity)
      return (vel[:,0], vel[:,1])