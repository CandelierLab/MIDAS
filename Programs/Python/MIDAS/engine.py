'''
MIDAS Engine
'''

import cmath
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

  def __init__(self, dimension, **kwargs):

    # --- Dimension

    self.dimension = dimension

    # --- Arena

    # Arena shape ('circular' or 'rectangular')
    self.arena = kwargs['arena'] if 'arena' in kwargs else 'rectangular'
    self.arena_shape =  kwargs['shape'] if 'shape' in kwargs else [1]*self.dimension

    # --- Boundary conditions

    match self.arena:
      case 'circular':
        self.periodic = kwargs['periodic'] if 'periodic' in kwargs else True
      case 'rectangular':
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
  
  def vel2ang(self, vel):
    '''
    Velocity to orientation conversion
    '''

    match self.dimension:
      case 1:
        pass
      case 2:
        ang = np.angle(vel[:,0] + 1j*vel[:,1])[:,None]
      case 3:
        pass

    return ang
  
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

  def __init__(self, dimension=2, **kwargs):
    '''
    Constructor

    Initializes the geometry and the agents
    '''

    # --- Initialization

    self.geom = Geometry(dimension, **kwargs)
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
    
    # --- Orientations

    ang = self.geom.vel2ang(vel)

    # --- Agents definition ------------------------------------------------

    self.agents.N_agents += N

    # Agent type    
    self.agents.atype = np.concatenate((self.agents.atype, 
                                        self.agents.atypes.index(gtype.lower())*np.ones(N, dtype=int)), axis=0)

    # Position and speed
    self.agents.pos = np.concatenate((self.agents.pos, pos), axis=0)
    self.agents.vel = np.concatenate((self.agents.vel, vel), axis=0)
    self.agents.ang = np.concatenate((self.agents.ang, ang), axis=0)
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

    self.window = Window('MIDAS', style=style)

    match self.geom.dimension:
      case 1:
        pass
      case 2:
        self.animation = MIDAS.animation.Animation2d(self)
      case 3:
        pass
    
    self.window.add(self.animation)

    # Forbid backward animation
    self.window.allow_backward = False

  # ------------------------------------------------------------------------
  #   Step
  # ------------------------------------------------------------------------

  def step(self, i):

    # print('--- Step', i, '-'*50)

    # print(self.agents.pos[0,:])

    # Double-buffer computation trick
    if i % 2:
      
      CUDA_step[self.cuda.gridDim, self.cuda.blockDim](i, self.cuda.atype,
        self.cuda.p0, self.cuda.v0, self.cuda.p1, self.cuda.v1,
        self.cuda.noise, self.cuda.group, self.cuda.param)
      
      cuda.synchronize()
      
      self.agents.pos = self.cuda.p1.copy_to_host()
      self.agents.vel = self.cuda.v1.copy_to_host()

    else:

      CUDA_step[self.cuda.gridDim, self.cuda.blockDim](i, self.cuda.atype,
        self.cuda.p1, self.cuda.v1, self.cuda.p0, self.cuda.v0,
        self.cuda.noise, self.cuda.group, self.cuda.param)
      
      cuda.synchronize()
      
      self.agents.pos = self.cuda.p0.copy_to_host()
      self.agents.vel = self.cuda.v0.copy_to_host()

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
    param[1] = self.geom.arena_shape[0]/2
    param[4] = self.geom.periodic if self.geom.arena=='circular' else self.geom.periodic[0]

    if self.geom.dimension>1:
      param[2] = self.geom.arena_shape[1]/2
      param[5] = self.geom.periodic if self.geom.arena=='circular' else self.geom.periodic[1]

    if self.geom.dimension>2:
      param[3] = self.geom.arena_shape[2]/2
      param[6] = self.geom.periodic if self.geom.arena=='circular' else self.geom.periodic[2]

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
    self.noise = None
    self.group = None
    
# --------------------------------------------------------------------------
#   The CUDA kernel
# --------------------------------------------------------------------------

@cuda.jit
def CUDA_step(i, atype, p0, v0, p1, v1, noise, group, param):
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

    # === Computation ======================================================

    # Polar coordinates
    v, alpha = cmath.polar(complex(v0[i,0],v0[i,1]))

    # --- Blind agents -----------------------------------------------------

    if atype[i]==1:
      da = 0

    # === Finalization =====================================================

    # --- Noise ------------------------------------------------------------

    # Speed noise
    vn = 0

    # Angular noise
    an = 0

    # --- Update -----------------------------------------------------------

    alpha += da + an
    z = cmath.rect(v, alpha)
    
    # Velocity
    v1[i,0] = z.real
    v1[i,1] = z.imag

    # Position
    p1[i,0] = p0[i,0] + v1[i,0]
    p1[i,1] = p0[i,1] + v1[i,1]

    # --- Boundary conditions

    if arena==0:
      '''
      Circular arena
      '''

      # Check for outsiders
      z1 = complex(p1[i,0], p1[i,1])
      if abs(z1) > arena_X:

        if periodic_X:
          z1 = cmath.rect(2*arena_X-abs(z1), cmath.phase(z1)+cmath.pi)
          p1[i,0] = z1.real
          p1[i,1] = z1.imag
          
        else:
          pass

      

    elif arena==1:  
      '''
      Rectangular arena
      '''

      # First dimension
      if periodic_X:
        if p1[i,0] > arena_X: p1[i,0] -= 2*arena_X
        if p1[i,0] < -arena_X: p1[i,0] += 2*arena_X
      else:
        if p1[i,0] > arena_X:
          p1[i,0] = 2*arena_X - p1[i,0]
          v1[i,0] = -v1[i,0]
        if p1[i,0] < -arena_X:
          p1[i,0] = -2*arena_X - p1[i,0]
          v1[i,0] = -v1[i,0]

      # Second dimension
      if dim>1:
        if periodic_Y:
          if p1[i,1] > arena_Y: p1[i,1] -= 2*arena_Y
          if p1[i,1] < -arena_Y: p1[i,1] += 2*arena_Y
        else:
          if p1[i,1] > arena_Y:
            p1[i,1] = 2*arena_Y - p1[i,1]
            v1[i,1] = -v1[i,1]
          if p1[i,1] < -arena_Y:
            p1[i,1] = -2*arena_Y - p1[i,1]
            v1[i,1] = -v1[i,1]

      # Third dimension
      if dim>2:
        if periodic_Z:
          if p1[i,2] > arena_Z: p1[i,2] -= 2*arena_Z
          if p1[i,2] < -arena_Z: p1[i,2] += 2*arena_Z
        else:
          if p1[i,2] > arena_Z:
            p1[i,2] = 2*arena_Z - p1[i,2]
            v1[i,2] = -v1[i,2]
          if p1[i,2] < -arena_Z:
            p1[i,2] = -2*arena_Z - p1[i,2]
            v1[i,2] = -v1[i,2]
    