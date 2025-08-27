'''
MIDAS core - Arena geometry
'''

import warnings
import numpy as np
from rich import print
from rich.panel import Panel

import MIDAS

# ══════════════════════════════════════════════════════════════════════════
class arena:

  # ────────────────────────────────────────────────────────────────────────
  def __init__(self, type, dimension, shape, periodic):

    self.type = type
    self.dimension = dimension

    # ─── Default values

    # Shape
    self.shape = np.array([1]*self.dimension if shape is None else shape)

    match self.type:

      case MIDAS.ARENA.RECTANGULAR:

        # ─── Boundary conditions

        if periodic is None:
          self.periodic = [True]*self.dimension

        elif isinstance(periodic, bool):
          self.periodic = [periodic for i in range(self.dimension)]

        else:
          self.periodic = periodic

      case MIDAS.ARENA.CIRCULAR:

        # Radius
        self.radius = self.shape[0]/2

        # ─── Boundary conditions

        '''
        NB: Periodic boundary conditions are not possible with a circular arena.
        Though coherent rules for a single agent are possible, it seems
        impossible to maintain a constant distance between two agents that are
        moving in parallel for instance, so distances are not conserved.
        '''

        if np.any(periodic):
          warnings.warn('Periodic boundary conditions are not possible with a circular arena. Switching to reflexive boundary conditions.')
        self.periodic = False

  # ────────────────────────────────────────────────────────────────────────
  def rich(self, title=True):
    
    s = ''

    if title:
      s += '[b i #abcdef]arena[/]\n'

    s += f'{self.dimension}d [cyan]'

    match self.type:
      case MIDAS.ARENA.RECTANGULAR: s += 'rectangular'
      case MIDAS.ARENA.CIRCULAR: s += 'circular'

    s += '[/] arena\n'

    match self.type:
      case MIDAS.ARENA.RECTANGULAR: s += f'shape:       {self.shape}\n'
      case MIDAS.ARENA.CIRCULAR: s += f'radius:       {self.radius}\n'
    
    s += f'periodicity: {self.periodic}'
    return s

  # ────────────────────────────────────────────────────────────────────────
  def display(self): print(Panel(self.rich(title=False), title='arena'))

# ══════════════════════════════════════════════════════════════════════════
class geometry:
  '''
  Geometry of the simulation, including:
  - Dimension
  - Arena type and shape ('circular' or 'rectangular')
  - Boundary conditions
  '''

  # ────────────────────────────────────────────────────────────────────────
  def __init__(self, dimension = 2, 
                     type = MIDAS.ARENA.RECTANGULAR,
                     shape = None,
                     periodic = None):

    # Dimension
    self.dimension = dimension

    # Arena
    self.arena = arena(type, self.dimension, shape, periodic)
    
  # ────────────────────────────────────────────────────────────────────────
  def rich(self):    
    return self.arena.rich()
  
  # ────────────────────────────────────────────────────────────────────────
  def display(self): 
    print(Panel(self.rich(), title="geometry"))

  # ────────────────────────────────────────────────────────────────────────
  def set_initial_positions(self, ptype, n):
    '''
    Initial positions
    '''

    if ptype in [None, 'random', 'shuffle']:

      # --- Random positions

      match self.arena:

        case MIDAS.ARENA.RECTANGULAR:

          pos = (np.random.rand(n, self.dimension)-1/2)
          for d in range(self.dimension):
            pos[:,d] *= self.arena_shape[d]

        case MIDAS.ARENA.CIRCULAR:

          # 2D
          match self.dimension:

            case 2:
              u1 = np.random.rand(n)
              u2 = np.random.rand(n)
              pos = np.column_stack((np.sqrt(u2)*np.cos(2*np.pi*u1),
                                     np.sqrt(u2)*np.sin(2*np.pi*u1)))*self.arena_shape[0]/2
          
            case _:

              # ------------------
              # !! TO IMPLEMENT !!
              # ------------------

              pos = (np.random.rand(n, self.dimension)-1/2)
              for d in range(self.dimension):
                pos[:,d] *= self.arena_shape[d]

    elif isinstance(ptype, list):
      
      match ptype[0]:
        
        case 'concentrated':
          ''' Concentration in a circle of given radius.'''

          match self.dimension:

            case 2:
              u1 = np.random.rand(n)
              u2 = np.random.rand(n)
              pos = np.column_stack((np.sqrt(u2)*np.cos(2*np.pi*u1),
                                     np.sqrt(u2)*np.sin(2*np.pi*u1)))*ptype[1]

        case 'condensed':
          ''' Condensed in a Gaussian density field of given size.'''

          match self.dimension:

            case 2:
              pos = np.random.randn(n,2)*ptype[1]

    return pos
  
  # ────────────────────────────────────────────────────────────────────────
  def set_initial_orientations(self, orientation, n):
    '''
    Initial velocities
    '''
      
    if orientation in [None, 'random', 'shuffle']:
      orientation = 2*np.pi*np.random.rand(n)

    return orientation