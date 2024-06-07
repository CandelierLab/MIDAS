'''
MIDAS Engine
'''

import math, cmath
import time
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32

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
  
  def set_initial_orientations(self, orientation, n):
    '''
    Initial velocities
    '''
      
    if orientation in [None, 'random', 'shuffle']:
      orientation = 2*np.pi*np.random.rand(n)

    return orientation
    
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
    '''
    Position are expressed in cartesian cooridnates (x,y,z)
    Velocities are expressed in polar coordinates (v,alpha,beta)
    '''
    self.pos = np.empty((0, dimension))
    self.vel = np.empty((0, dimension))
    self.vlim = np.empty((0, 2))

    # Noise
    '''
      0: velocity amplitude noise
      (1,2): angular noises
    '''
    self.noise = np.empty((0, dimension))

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
      initial_condition = {'position': None, 'orientation': None, 'speed': 0.01}

    # --- Positions

    if type(initial_condition['position']) in [type(None), str]:
      pos = self.geom.set_initial_positions(initial_condition['position'], N)
    else:
      pos = np.array(initial_condition['position'])

    # --- Velocities

    # Speed vector
    V = initial_condition['speed']*np.ones(N) if type(initial_condition['speed']) in [int, float] else initial_condition['speed']

    if type(initial_condition['orientation']) in [type(None), str]:
      alpha = self.geom.set_initial_orientations(initial_condition['orientation'], N)
    else:
      alpha = np.array(initial_condition['orientation'])
    vel = np.column_stack((V, alpha))
    
    # Limits
    vlim = np.ones((N,2), dtype=np.float32)
    vlim[:,0] = kwargs['vmin'] if 'vmin' in kwargs else 0
    vlim[:,1] = kwargs['vmax'] if 'vmax' in kwargs else V

    # --- Noise

    noise = np.ones((N,2), dtype=np.float32)
    noise[:,0] = kwargs['vnoise'] if 'vnoise' in kwargs else 0
    noise[:,1] = kwargs['anoise'] if 'anoise' in kwargs else 0

    # --- Agents definition ------------------------------------------------

    self.agents.N_agents += N

    # Agent type    
    self.agents.atype = np.concatenate((self.agents.atype, 
                                        self.agents.atypes.index(gtype.lower())*np.ones(N, dtype=int)), axis=0)

    # Position and speed
    self.agents.pos = np.concatenate((self.agents.pos, pos), axis=0)
    self.agents.vel = np.concatenate((self.agents.vel, vel), axis=0)
    self.agents.vlim = np.concatenate((self.agents.vlim, vlim), axis=0)

    # Noise
    self.agents.noise = np.concatenate((self.agents.noise, noise), axis=0)

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

    # Double-buffer computation trick
    if i % 2:
      
      CUDA_step[self.cuda.gridDim, self.cuda.blockDim](i, self.cuda.atype, self.cuda.group,
        self.cuda.p0, self.cuda.v0, self.cuda.p1, self.cuda.v1,
        self.cuda.noise, self.cuda.vlim, self.cuda.param, self.cuda.rng)
      
      cuda.synchronize()
      
      self.agents.pos = self.cuda.p1.copy_to_host()
      self.agents.vel = self.cuda.v1.copy_to_host()

    else:

      CUDA_step[self.cuda.gridDim, self.cuda.blockDim](i, self.cuda.atype, self.cuda.group,
        self.cuda.p1, self.cuda.v1, self.cuda.p0, self.cuda.v0,
        self.cuda.noise, self.cuda.vlim, self.cuda.param, self.cuda.rng)
      
      cuda.synchronize()
      
      self.agents.pos = self.cuda.p0.copy_to_host()
      self.agents.vel = self.cuda.v0.copy_to_host()
    
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

    # Random number generator
    self.cuda.rng = create_xoroshiro128p_states(self.cuda.blockDim*self.cuda.gridDim, seed=0)

    # Send arrays to device
    self.cuda.atype = cuda.to_device(self.agents.atype.astype(np.float32))
    self.cuda.p0 = cuda.to_device(self.agents.pos.astype(np.float32))
    self.cuda.v0 = cuda.to_device(self.agents.vel.astype(np.float32))
    self.cuda.vlim = cuda.to_device(self.agents.vlim.astype(np.float32))
    self.cuda.noise = cuda.to_device(self.agents.noise.astype(np.float32))
    self.cuda.group = cuda.to_device(self.agents.group.astype(np.float32))

    # Double buffers
    self.cuda.p1 = cuda.device_array((self.agents.N_agents, self.geom.dimension), np.float32)
    self.cuda.v1 = cuda.device_array((self.agents.N_agents, self.geom.dimension), np.float32)

    # --- Parameter serialization ------------------------------------------

    match self.geom.dimension:

      case 1:

        param = np.zeros(3, dtype=np.float32)

        # Arena shape
        param[0] = 0 if self.geom.arena=='circular' else 1

        # Arena size
        param[1] = self.geom.arena_shape[0]/2

        # Arena periodicity
        param[2] = self.geom.periodic if self.geom.arena=='circular' else self.geom.periodic[0]

      case 2:

        param = np.zeros(5, dtype=np.float32)

        # Arena shape
        param[0] = 0 if self.geom.arena=='circular' else 1

        # Arena size
        param[1] = self.geom.arena_shape[0]/2
        param[2] = self.geom.arena_shape[1]/2

        # Arena periodicity
        param[3] = self.geom.periodic if self.geom.arena=='circular' else self.geom.periodic[0]
        param[4] = self.geom.periodic if self.geom.arena=='circular' else self.geom.periodic[1]

      case 3:

        param = np.zeros(7, dtype=np.float32)

        # Arena shape
        param[0] = 0 if self.geom.arena=='circular' else 1

        # Arena size
        param[1] = self.geom.arena_shape[0]/2
        param[2] = self.geom.arena_shape[1]/2
        param[3] = self.geom.arena_shape[2]/2

        # Arena periodicity
        param[4] = self.geom.periodic if self.geom.arena=='circular' else self.geom.periodic[0]
        param[5] = self.geom.periodic if self.geom.arena=='circular' else self.geom.periodic[1]
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
    self.group = None
    self.noise = None
    self.vlim = None

    # Random number generator
    self.rng = None
    
# --------------------------------------------------------------------------
#   The CUDA kernel
# --------------------------------------------------------------------------

@cuda.jit
def CUDA_step(i, atype, group, p0, v0, p1, v1, noise, vlim, param, rng):
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

    '''
    arena:
      0: circular
      1: rectangular
    arena_X,Y,Z:
      circular arena: radius
      rectangular arena: width/2, height/2, depth/2
    periodic_X,Y,Z:
      0: reflexive
      1: periodic
    '''

    # Arena        
    arena = param[0]

    match dim:

      case 1:

        # Arena shape
        arena_X = param[1]

        # Arena periodicity
        periodic_X = param[2]

      case 2:

        # Arena shape
        arena_X = param[1]
        arena_Y = param[2]

        # Arena periodicity
        periodic_X = param[3]
        periodic_Y = param[4]

      case 3:

        # Arena shape
        arena_X = param[1]
        arena_Y = param[2]
        arena_Z = param[3]

        # Arena periodicity
        periodic_X = param[4]
        periodic_Y = param[5]
        periodic_Z = param[6]

    # === Computation ======================================================

    # Polar coordinates
    v = v0[i,0]
    alpha = v0[i,1]

    # --- Blind agents -----------------------------------------------------

    if atype[i]==1:
      da = 0

    # ... 

    # === Update ===========================================================

    match dim:

      case 2:

        # Reorientation
        alpha += da

        # --- Noise --------------------------------------------------------

        # Speed noise
        if noise[i,0]:
          v += noise[i,0]*xoroshiro128p_normal_float32(rng, i)
          if v < vlim[i,0]: v = vlim[i,0]
          elif v > vlim[i,1]: v = vlim[i,1]

        # Angular noise
        if noise[i,1]:
          alpha += noise[i,1]*xoroshiro128p_normal_float32(rng, i)

        # Candidate position and velocity
        z0 = complex(p0[i,0], p0[i,1])
        z1 = z0 + cmath.rect(v, alpha)

        # Boundary conditions
        p1[i,0], p1[i,1], v1[i,0], v1[i,1] = assign_2d(z0, z1, v, alpha, arena,
                                                       arena_X, arena_Y,
                                                       periodic_X, periodic_Y)

# --------------------------------------------------------------------------
#   Device functions
# --------------------------------------------------------------------------

@cuda.jit(device=True)
def assign_2d(z0, z1, v, alpha, arena, arena_X, arena_Y, periodic_X, periodic_Y):
  
  if v==0:
    return (z1.real, z1.imag, v, alpha)

  if arena==0:
    '''
    Circular arena
    '''

    # Check for outsiders
    if abs(z1) > arena_X:

      if periodic_X:
        '''
        Periodic circular
        '''
        z1 = cmath.rect(2*arena_X-abs(z1), cmath.phase(z1)+cmath.pi)
        
      else:
        '''
        Reflexive circular
        '''

        # Crossing point
        phi = alpha + math.asin((z0.imag*math.cos(alpha) - z0.real*math.sin(alpha))/arena_X)
        zc = cmath.rect(arena_X, phi)

        # Position        
        z1 = zc + (v-abs(zc-z0))*cmath.exp(1j*(cmath.pi + 2*phi - alpha))

        # Final velocity
        alpha += cmath.pi-2*(alpha-phi)

    # Finam position
    px = z1.real
    py = z1.imag


  elif arena==1:  
    '''
    Rectangular arena
    '''

    zv = cmath.rect(v, alpha)
    vx = zv.real
    vy = zv.imag

    # First dimension
    if periodic_X:

      if z1.real > arena_X: px = z1.real - 2*arena_X
      elif z1.real < -arena_X: px = z1.real + 2*arena_X
      else: px = z1.real

    else:

      if z1.real > arena_X:
        px = 2*arena_X - z1.real
        vx = -zv.real
      elif z1.real < -arena_X:
        px = -2*arena_X - z1.real
        vx = -zv.real
      else:
        px = z1.real

    # Second dimension
    if periodic_Y:

      if z1.imag > arena_Y: py = z1.imag - 2*arena_Y
      elif z1.imag < -arena_Y: py = z1.imag + 2*arena_Y
      else: py = z1.imag

    else:

      if z1.imag > arena_Y:
        py = 2*arena_Y - z1.imag
        vy = -zv.imag
      elif z1.imag < -arena_Y:
        py = -2*arena_Y - z1.imag
        vy = -zv.imag
      else:
        py = z1.imag

    v, alpha = cmath.polar(complex(vx, vy))

  return (px, py, v, alpha)