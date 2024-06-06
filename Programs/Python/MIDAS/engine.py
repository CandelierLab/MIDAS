'''
MIDAS Engine
'''

import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt

from Animation.Window import Window
import MIDAS.animation

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

    # Types
    '''
    Agent atypes are:
    0: Fixed
    1: Blind
    2: RIPO
    3: RINNO
    '''
    self.atypes = ['fixed', 'blind', 'ripo', 'rinno']
    self.atype = np.empty(0, dtype=int)

    # Positions and velocities
    self.pos = np.empty((0, dimension))
    self.vel = np.empty((0, dimension))
    self.ang = np.empty((0, dimension-1))
    self.speed = np.empty(0)

    # Angular noise
    self.noise = np.empty(0)

    # Groups
    self.N_groups = 0    
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
    self.window = None
    self.animation = None

    # GPU
    self.cuda = CUDA()

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

    # Agent type    
    self.agents.atype = np.concatenate((self.agents.atype, 
                                        self.agents.atypes.index(gtype.lower())*np.ones(N, dtype=int)), axis=0)

    # Position and speed
    self.agents.pos = np.concatenate((self.agents.pos, pos), axis=0)
    self.agents.vel = np.concatenate((self.agents.vel, vel), axis=0)
    self.agents.speed = np.concatenate((self.agents.speed, speed), axis=0)

    # Groups
    if gname in self.agents.group_names:
      iname = self.agents.group_names.index(gname)
    else:
      iname = len(self.agents.group_names)
      self.agents.group_names.append(gname)
      self.agents.N_groups += 1

    self.agents.group = np.concatenate((self.agents.group, iname*np.ones(N, dtype=int)), axis=0)

  # ------------------------------------------------------------------------
  #   Animation setup
  # ------------------------------------------------------------------------

  def setup_animation(self, style='dark'):
    '''
    Define animation
    '''

    self.window = Window('Simple animation', style=style)
    self.animation = MIDAS.animation.Animation(self)
    
    self.window.add(self.animation)

    # Forbid backward animation
    self.window.allow_backward = False


  # ------------------------------------------------------------------------
  #   Step
  # ------------------------------------------------------------------------

  def step(self, i):

    print('--- Step', i, '-'*50)

    # print(self.agents.pos[0,:])

    # Double-buffer computation trick
    if i % 2:
      
      CUDA_step[self.cuda.gridDim, self.cuda.blockDim](i, self.cuda.atype,
        self.cuda.p0, self.cuda.v0, self.cuda.p1, self.cuda.v1,
        self.cuda.speed, self.cuda.noise, self.cuda.group, self.cuda.param)
      
      cuda.synchronize()
      
      self.agents.pos = self.cuda.p1.copy_to_host()

    else:

      CUDA_step[self.cuda.gridDim, self.cuda.blockDim](i, self.cuda.atype,
        self.cuda.p1, self.cuda.v1, self.cuda.p0, self.cuda.v0,
        self.cuda.speed, self.cuda.noise, self.cuda.group, self.cuda.param)
      
      cuda.synchronize()
      
      self.agents.pos = self.cuda.p0.copy_to_host()

    # print(self.agents.pos[0,:])
    
  # ------------------------------------------------------------------------
  #   Run
  # ------------------------------------------------------------------------

  def run(self):

    print(f'Running simulation with {self.steps} steps ...')

    # Reference time
    self.tref = time.time()

    # --- CUDA preparation -------------------------------------------------
    
    # Threads and blocks
    self.cuda.blockDim = 32
    self.cuda.gridDim = (self.agents.N_agents + (self.cuda.blockDim - 1)) // self.cuda.blockDim

    # Send arrays to device
    self.cuda.atype = cuda.to_device(self.agents.atype.astype(np.float32))
    self.cuda.p0 = cuda.to_device(self.agents.pos.astype(np.float32))
    self.cuda.v0 = cuda.to_device(self.agents.vel.astype(np.float32))
    self.cuda.speed = cuda.to_device(self.agents.speed.astype(np.float32))
    self.cuda.noise = cuda.to_device(self.agents.noise.astype(np.float32))
    self.cuda.group = cuda.to_device(self.agents.group.astype(np.float32))

    # Double buffers
    self.cuda.p1 = cuda.device_array((self.agents.N_agents, self.geom.dimension), np.float32)
    self.cuda.v1 = cuda.device_array((self.agents.N_agents, self.geom.dimension), np.float32)

    # --- Parameter serialization

    param = np.zeros(7, dtype=np.float32)

    # Arena shape
    match self.geom.arena:
      case 'circular': param[0] = 0
      case 'rectangular': param[0] = 1

    # Arena size and periodicity
    param[1] = self.geom.arena_shape[0]
    param[4] = self.geom.periodic[0]

    if self.geom.dimension>1:
      param[2] = self.geom.arena_shape[1]
      param[5] = self.geom.periodic[1]

    if self.geom.dimension>2:
      param[3] = self.geom.arena_shape[2]
      param[6] = self.geom.periodic[2]

    self.cuda.param = cuda.to_device(param.astype(np.float32))

    # --- Main loop --------------------------------------------------------

    if self.animation is None:
      i = 0
      while self.steps is None or i<self.steps:
        self.step(i)
        i += 1

    else:

      # Use the animation clock
      self.animation.initialize()
      self.window.show()

# === CUDA ===============================================================

class CUDA:

  def __init__(self):

    self.blockDim = None
    self.gridDim = None

    # Algorithm parameters
    self.param = None

    # Double buffers
    self.p0 = None
    self.v0 = None
    self.p1 = None
    self.v1 = None

    # Other required arrays
    self.atype = None
    self.speed = None
    self.noise = None
    self.group = None
    
# --------------------------------------------------------------------------
#   The CUDA kernel
# --------------------------------------------------------------------------

@cuda.jit
def CUDA_step(i, atype, p0, v0, p1, v1, speed, noise, group, param):
  '''
  The CUDA kernel
  '''

  i = cuda.grid(1)

  if i<p0.shape[0]:

    N, dim = p0.shape

    # --- Fixed points -----------------------------------------------------

    if atype[i]==0:
      for j in range(dim):
        p1[i,j] = p0[i,j]

    # --- Deserialization of the parameters --------------------------------

    # Arena
    '''
    0: circular
    1: rectangular
    '''
    arena = param[0]

    # Arena shape
    arena_X = param[1]
    arena_Y = param[2]
    arena_Z = param[3]

    # Arena periodicity
    periodic_X = param[4]
    periodic_Y = param[5]
    periodic_Z = param[6]

    # if i==0:
    #   # print(N, dim, arena, arena_X, periodic_X)
    #   print(p0[i,0])

    # --- Computation ------------------------------------------------------

    # Blind agents
    for j in range(dim):
      p1[i,j] = p0[i,j] + v0[i,j]
      v1[i,j] = v0[i,j]