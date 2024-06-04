'''
MIDAS Engine
'''

import time
import numpy as np
import matplotlib.pyplot as plt

# === GEOMETRY =============================================================

class Geometry:
  '''
  Geometry of the simulation, including:
  - Dimension
  - Arena type and shape
  - Boundary conditions
  '''

  def __init__(self, dimension, arena_type='rectangular', **kwargs):

    # --- Dimension

    self.dimension = dimension

    # --- Arena

    # Arena shape ('circular' or 'rectangular')
    self.arena = arena_type
    self.arena_shape =  kwargs['shape'] if 'shape' in kwargs else [1]*self.dimension

    # --- Boundary conditions

    self.periodic = kwargs['periodic'] if 'periodic' in kwargs else [True]*self.dimension
  
  def set_initial_positions(self, ptype, n):
    '''
    Initial positions
    '''

    if ptype in [None, 'random', 'shuffle']:

      # --- Random positions

      match self.arena:

        case 'rectangular':

          pos = (np.random.rand(n, self.dimension)-1/2)
          for d in range(self.dimension):
            pos[:,d] *= self.arena_shape[d]

        case 'circular':

          # TO DO
          pass
      
    return pos
  
  def set_initial_velocities(self, ptype, n, speeds):
    '''
    Initial velocities
    '''
      
    if ptype in [None, 'random', 'shuffle']:

      # --- Random velocities

      tmp = np.random.default_rng().multivariate_normal(np.zeros(self.dimension), np.eye(self.dimension), size=n)
      vel = speeds[:,None]*tmp/np.sqrt(np.sum(tmp**2, axis=1))[:,None]

    return vel

# === AGENTS ===============================================================

class Agents:
  '''
  Collection of all agents
  '''

  def __init__(self, dimension):
  
    self.N_agents = 0

    # Positions and velocities
    self.pos = np.empty((0, dimension))
    self.vel = np.empty((0, dimension))
    self.speed = np.empty(0)

    # Groups
    self.N_groups = 0
    self.group_types = []
    self.group_names = []
    self.group = np.empty(0, dtype=int)

# === ENGINE ===============================================================

class Engine:
  '''
  Engine
  '''

  # ------------------------------------------------------------------------
  #   Contructor
  # ------------------------------------------------------------------------

  def __init__(self, dimension=2):
    '''
    Constructor

    Initializes the geometry and the agents
    '''

    # --- Initialization

    self.geom = Geometry(dimension)
    self.agents = Agents(dimension)

    # Associated animation
    self.animation = None

    # --- Time

    # Total number of steps
    self.steps = 1

    # Computation time reference
    self.tref = None

  # ------------------------------------------------------------------------
  #   Add group
  # ------------------------------------------------------------------------

  def add_group(self, gtype, N, **kwargs):

    # Group name
    gname = kwargs['name'] if 'name' in kwargs else gtype

    # --- Initial conditions -----------------------------------------------

    # --- User definition

    if 'initial_condition' in kwargs:
      initial_condition = kwargs['IC']
    else:
      initial_condition = {'position': None, 'velocity': None, 'speed': 0.01}

    # --- Positions

    if type(initial_condition['position']) in [type(None), str]:
      pos = self.geom.set_initial_positions(initial_condition['position'], N)
    else:
      pos = np.array(initial_condition['position'])

    # --- Velocities

    # Speed vector
    speed = initial_condition['speed']*np.ones(N) if type(initial_condition['speed']) in [int, float] else initial_condition['speed']

    if type(initial_condition['velocity']) in [type(None), str]:
      vel = self.geom.set_initial_velocities(initial_condition['velocity'], N, speed)
    else:
      vel = np.array(initial_condition['velocity'])
    
    # --- Agents definition ------------------------------------------------

    self.agents.N_agents += N

    # Position and speed
    self.agents.pos = np.concatenate((self.agents.pos, pos), axis=0)
    self.agents.vel = np.concatenate((self.agents.vel, vel), axis=0)
    self.agents.speed = np.concatenate((self.agents.speed, speed), axis=0)

    # Groups
    if gtype in self.agents.group_types:
      itype = self.agents.group_types.index(gtype)
    else:
      itype = len(self.agents.group_types)      
      self.agents.group_types.append(gtype)
      self.agents.group_names.append(gname)
      self.agents.N_groups += 1

    self.agents.group = np.concatenate((self.agents.group, itype*np.ones(N, dtype=int)), axis=0)

  # ------------------------------------------------------------------------
  #   Step
  # ------------------------------------------------------------------------

  def step(self, i):

    print(i)

    pass

  # ------------------------------------------------------------------------
  #   Run
  # ------------------------------------------------------------------------

  def run(self):

    print(f'Running simulation with {self.steps} steps ...')

    # Reference time
    self.tref = time.time()

    # --- Send arrays to device



    # --- Main loop --------------------------------------------------------

    if self.animation is None:
      i = 0
      while self.steps is None or i<self.steps:
        self.step(i)
        i += 1

    else:

      # Use the animation clock

      pass
      # self.animation.initialize()
      # self.window.show()



