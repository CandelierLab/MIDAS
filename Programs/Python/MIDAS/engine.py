'''
MIDAS Engine
'''

import warnings
import math, cmath
import time
import numpy as np
import numba as nb
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32

from Animation.Window import Window

from MIDAS.enums import *
import MIDAS.animation

# import sys
# sys.path.append("/home/raphael/Science/Projects/Toolboxes/MIDAS/Programs/Python/test/")
# import RIPO

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
    self.arena = kwargs['arena'] if 'arena' in kwargs else Arena.RECTANGULAR
    self.arena_shape =  kwargs['shape'] if 'shape' in kwargs else [1]*self.dimension

    # --- Boundary conditions

    match self.arena:
      case Arena.CIRCULAR:
        '''
        NB: Periodic boundary conditions are not possible with a circular arena.
        Though coherent rules for a single agent are possible, it seems
        impossible to maintain a constant distance between two agents that are
        moving in parallel for instance, so distances are not conserved.
        '''
        if 'periodic' in kwargs and kwargs['periodic']:
          warnings.warn('Periodic boundary conditions are not possible with a circular arena. Switching to reflexive boundary conditions.')
        self.periodic = False

      case Arena.RECTANGULAR:
        self.periodic = kwargs['periodic'] if 'periodic' in kwargs else [True]*self.dimension
  
  def set_initial_positions(self, ptype, n):
    '''
    Initial positions
    '''

    if ptype in [None, 'random', 'shuffle']:

      # --- Random positions

      match self.arena:

        case Arena.RECTANGULAR:

          pos = (np.random.rand(n, self.dimension)-1/2)
          for d in range(self.dimension):
            pos[:,d] *= self.arena_shape[d]

        case Arena.CIRCULAR:

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
  
    # Number of agents
    self.N = 0

    # Types
    self.atype = np.empty(0, dtype=int)

    # Positions and velocities
    '''
    Position are expressed in cartesian coordinates (x,y,z)
    Velocities are expressed in polar coordinates (v,alpha,beta)
    '''
    self.pos = np.empty((0, dimension))
    self.vel = np.empty((0, dimension))

    # Agent parameters
    self.param = np.empty((0, dimension+2))

    # Groups
    self.group = np.empty(0, dtype=int)

# === GROUPS ===============================================================

class Groups:
  '''
  Subcollection of agents
  
  NB: All agents in a group have the same type
  '''

  def __init__(self, dimension):
  
    self.dimension = dimension

    # Number of groups
    self.N = 0

    # Group names
    self.names = []

    # Types of the agents in each group
    self.atype = np.empty(0, dtype=int)
    
    # Group parameters
    self.param = np.empty(0, dtype=np.float32)

  # ------------------------------------------------------------------------
  #   Parameter serialization
  # ------------------------------------------------------------------------

  def add_RIPO(self, **kwargs):

    param = []

    # --- Radii of sectors

    if 'rS' in kwargs:

      rS = np.sort(kwargs['rS'])
      nR =  rS.size + 1 

      param.append(nR)
      [param.append(x) for x in rS]

    else:
      param.append(1)

    # --- Maximal radius

    if 'rmax' in kwargs and kwargs['rmax'] is not None:
      param.append(kwargs['rmax'])
    else:
      param.append(0)

    # --- Angular slices

    if self.dimension>1:
      nSa = kwargs['nSa'] if 'nSa' in kwargs else 4
      param.append(nSa)
    else:
      nSa = 1

    if self.dimension>2:
      nSb = kwargs['nSb'] if 'nSb' in kwargs else 4
      param.append(nSb)
    else:
      nSb = 1

    # --- Coefficients    
    '''
    Each grid is composed of nS = nR*nSa*nSb sector.
    The number of coefficients depends on the input types:
    - Walls: nS
    - Field: nS.nF
    - Group: nS.nG^2
    '''

    for k, C in kwargs['coefficients'].items():
      param.append(k.value)
      [param.append(c) for c in C]
   
    # Convert to numpy
    param = np.array(param, dtype=np.float32)

    if self.param.size:
      if self.param.shape[1]>param.size:
        param = np.pad(param, (0,self.param.shape[1]-param.size))
      elif self.param.shape[1]<param.size:
        self.param = np.pad(self.param, ((0,0),(0,param.size-self.param.shape[1])))
        
        self.param = np.concatenate((self.param, param[None,:]), axis=0)
    else:
      self.param = param[None,:]

    # print(self.param)

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
    self.groups = Groups(dimension)
    
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

    # --- Misc attributes
    
    self.verbose = True

  # ------------------------------------------------------------------------
  #   Add group
  # ------------------------------------------------------------------------

  def add_group(self, gtype, N, **kwargs):

    # Group name
    gname = kwargs['name'] if 'name' in kwargs else gtype.name

    # --- Initial conditions -----------------------------------------------

    # --- User definition

    if 'initial_condition' in kwargs:
      initial_condition = kwargs['initial_condition']
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

    self.agents.N += N

    # Agent type    
    self.agents.atype = np.concatenate((self.agents.atype, 
                                        gtype.value*np.ones(N, dtype=int)), axis=0)

    # Position and speed
    self.agents.pos = np.concatenate((self.agents.pos, pos), axis=0)
    self.agents.vel = np.concatenate((self.agents.vel, vel), axis=0)

    # --- Other agent parameters ---

    # Speed and noise
    aparam = np.concatenate((vlim, noise), axis=1)

    # Agent parameters
    self.agents.param = np.concatenate((self.agents.param, aparam), axis=0)

    # --- Group definition -------------------------------------------------
    
    match gtype:
      case Agent.RIPO:
        self.groups.add_RIPO(**kwargs)

    # Groups
    if gname in self.groups.names:
      iname = self.groups.names.index(gname)
    else:
      iname = len(self.groups.names)
      self.groups.names.append(gname)
      self.groups.N += 1

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
      
      CUDA_step[self.cuda.gridDim, self.cuda.blockDim](self.cuda.geom, i, self.cuda.atype, self.cuda.group,
        self.cuda.p0, self.cuda.v0, self.cuda.p1, self.cuda.v1,
        self.cuda.aparam, self.cuda.gparam, self.cuda.rng)
      
      cuda.synchronize()
      
      self.agents.pos = self.cuda.p1.copy_to_host()
      self.agents.vel = self.cuda.v1.copy_to_host()

    else:

      CUDA_step[self.cuda.gridDim, self.cuda.blockDim](self.cuda.geom, i, self.cuda.atype, self.cuda.group,
        self.cuda.p1, self.cuda.v1, self.cuda.p0, self.cuda.v0,
        self.cuda.aparam, self.cuda.gparam, self.cuda.rng)
      
      cuda.synchronize()
      
      self.agents.pos = self.cuda.p0.copy_to_host()
      self.agents.vel = self.cuda.v0.copy_to_host()
    
    # --- End of simulation

    if self.steps is not None and i==self.steps-1:

      # End of simulation
      if self.verbose:
        print('→ End of simulation @ {:d} steps ({:.2f} s)'.format(self.steps, time.time()-self.tref))

      # End display
      if self.animation is not None:
        self.animation.window.close()

  # ------------------------------------------------------------------------
  #   Run
  # ------------------------------------------------------------------------

  def run(self):

    if self.verbose:
      print('-'*50)
      print(f'Running simulation with {self.steps} steps ...')

      print(self.groups.param)

    # Reference time
    self.tref = time.time()

    # --- CUDA preparation -------------------------------------------------
    
    # Threads and blocks
    self.cuda.blockDim = 32
    self.cuda.gridDim = (self.agents.N + (self.cuda.blockDim - 1)) // self.cuda.blockDim

    # Random number generator
    self.cuda.rng = create_xoroshiro128p_states(self.cuda.blockDim*self.cuda.gridDim, seed=0)

    # Send arrays to device
    self.cuda.atype = cuda.to_device(self.agents.atype.astype(np.float32))
    self.cuda.group = cuda.to_device(self.agents.group.astype(np.float32))
    self.cuda.p0 = cuda.to_device(self.agents.pos.astype(np.float32))
    self.cuda.v0 = cuda.to_device(self.agents.vel.astype(np.float32))
    self.cuda.aparam = cuda.to_device(self.agents.param.astype(np.float32))
    self.cuda.gparam = cuda.to_device(self.groups.param.astype(np.float32))

    # Double buffers
    self.cuda.p1 = cuda.device_array((self.agents.N, self.geom.dimension), np.float32)
    self.cuda.v1 = cuda.device_array((self.agents.N, self.geom.dimension), np.float32)

    # --- Parameter serialization ------------------------------------------

    match self.geom.dimension:

      case 1:

        geom = np.zeros(3, dtype=np.float32)

        # Arena shape
        geom[0] = self.geom.arena.value

        # Arena size
        geom[1] = self.geom.arena_shape[0]/2

        # Arena periodicity
        geom[2] = self.geom.periodic if self.geom.arena==Arena.CIRCULAR else self.geom.periodic[0]

      case 2:

        geom = np.zeros(5, dtype=np.float32)

        # Arena shape
        geom[0] = self.geom.arena.value

        # Arena size
        geom[1] = self.geom.arena_shape[0]/2
        geom[2] = self.geom.arena_shape[1]/2

        # Arena periodicity
        geom[3] = self.geom.periodic if self.geom.arena==Arena.CIRCULAR else self.geom.periodic[0]
        geom[4] = self.geom.periodic if self.geom.arena==Arena.CIRCULAR else self.geom.periodic[1]

      case 3:

        geom = np.zeros(7, dtype=np.float32)

        # Arena shape
        geom[0] = self.geom.arena.value

        # Arena size
        geom[1] = self.geom.arena_shape[0]/2
        geom[2] = self.geom.arena_shape[1]/2
        geom[3] = self.geom.arena_shape[2]/2

        # Arena periodicity
        geom[4] = self.geom.periodic if self.geom.arena==Arena.CIRCULAR else self.geom.periodic[0]
        geom[5] = self.geom.periodic if self.geom.arena==Arena.CIRCULAR else self.geom.periodic[1]
        geom[6] = self.geom.periodic if self.geom.arena==Arena.CIRCULAR else self.geom.periodic[2]

    self.cuda.geom = cuda.to_device(geom.astype(np.float32))

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

# === CUDA =================================================================

class CUDA:

  def __init__(self):

    self.blockDim = None
    self.gridDim = None

    # Geometric parameters
    self.geom = None

    # Double buffers
    self.p0 = None
    self.v0 = None
    self.p1 = None
    self.v1 = None

    # Other required arrays
    self.atype = None
    self.group = None
    self.aparam = None
    self.gparam = None
    
    # Radial Input specifics
    self.rS = None

    # Random number generator
    self.rng = None

############################################################################
############################################################################
# #                                                                      # #
# #                                                                      # #
# #                        DEVICE FUNCTIONS                              # #
# #                                                                      # #
# #                                                                      # #
############################################################################
############################################################################

# --------------------------------------------------------------------------
#   The CUDA kernel
# --------------------------------------------------------------------------

@cuda.jit
def CUDA_step(geom, i, atype, group, p0, v0, p1, v1, aparam, gparam, rng):
  '''
  The CUDA kernel
  '''

  i = cuda.grid(1)

  if i<p0.shape[0]:

    N, dim = p0.shape

    # === Fixed points =====================================================

    if atype[i]==Agent.FIXED.value:
      for j in range(dim):
        p1[i,j] = p0[i,j]

    # === Deserialization of the parameters ================================

    # --- Geometric parameters ---------------------------------------------
    '''
    arena_X,Y,Z:
      circular arena: radius
      rectangular arena: width/2, height/2, depth/2
    periodic_X,Y,Z:
      0: reflexive
      1: periodic
    '''

    # Arena        
    arena = geom[0]

    match dim:

      case 1:

        # Arena shape
        arena_X = geom[1]

        # Arena periodicity
        periodic_X = geom[2]

      case 2:

        # Arena shape
        arena_X = geom[1]
        arena_Y = geom[2]

        # Arena periodicity
        periodic_X = geom[3]
        periodic_Y = geom[4]

      case 3:

        # Arena shape
        arena_X = geom[1]
        arena_Y = geom[2]
        arena_Z = geom[3]

        # Arena periodicity
        periodic_X = geom[4]
        periodic_Y = geom[5]
        periodic_Z = geom[6]

    # --- Agent parameters -------------------------------------------------
    '''
    Agents parameters 
    ├── vlim: speed limits (size=2)
    ├── Noise: speed, alpha, beta (size=dim)
    ├── ... (see below for parameters based on each agent type)
    '''

    # Velocity limits
    vmin = aparam[i,0]
    vmax = aparam[i,1]

    # Noise
    vnoise = aparam[i,2]
    if dim>1: anoise = aparam[i,3]
    if dim>2: bnoise = aparam[i,4]

    # === Computation ======================================================

    # Polar coordinates
    v = v0[i,0]
    a = v0[i,1]

    # --- Blind agents -----------------------------------------------------

    if atype[i]==Agent.BLIND.value:
      dv = 0
      da = 0

    # --- RIPO agents ------------------------------------------------------

    if atype[i]==Agent.RIPO.value:

      # Group id
      gid = int(group[i])

      # === Deserialization of the RIPO parameters ===
      '''
      RIPO
      ├── nR          (1)
      ├── rS          (nR-1)
      ├── nSa         (1, if dim>1) 
      ├── nSb         (1, if dim>2)
      ├── Input type  (1)
      ├── coeffs      (var)
      ├── Output      (?)
      '''

      # --- Radial limits

      # Number of sectors per slice
      nR = int(gparam[gid, 0])

      # Sectors' radii first element
      rS0 = 1 if nR>1 else None

      # Maximal radius
      rmax = gparam[gid, nR] if gparam[gid, nR]>0 else None

      # --- Angular slices

      nSa = gparam[gid, nR+1] if dim>1 else 1
      nSb = gparam[gid, nR+2] if dim>2 else 1

      k = nR + dim

      # === Interactions ===================================================

      for j in range(N):

        # Skip self-perception
        if i==j: continue

        # Distance and relative orientation
        z, alpha, status = relative_2d(p0[i,0], p0[i,1], v0[i,1], p0[j,0], p0[j,1], v0[j,1], rmax, arena, arena_X, arena_Y, periodic_X, periodic_Y)

        if not status:
          print('None')
        else:
          print(z.real, z.imag, alpha)

      # a=cuda.local.array(shape=1,dtype=numba.float64)
      
      # === Inputs ===

      IN = 0

      # === Processing ===

      da = 0
      dv = 0

      # dv, da = RIPO_2d(i, p0, v0)
      pass      

    # === Update ===========================================================

    match dim:

      case 2:

        # Update velocity
        v += dv
        a += da

        # --- Noise --------------------------------------------------------

        # Speed noise
        if vnoise:
          v += vnoise*xoroshiro128p_normal_float32(rng, i)
          if v < vmin: v = vmin
          elif v > vmax: v = vmax

        # Angular noise
        if anoise:
          a += anoise*xoroshiro128p_normal_float32(rng, i)

        # Candidate position and velocity
        z0 = complex(p0[i,0], p0[i,1])
        z1 = z0 + cmath.rect(v, a)

        # Boundary conditions
        p1[i,0], p1[i,1], v1[i,0], v1[i,1] = assign_2d(z0, z1, v, a, arena,
                                                       arena_X, arena_Y,
                                                       periodic_X, periodic_Y)

# --------------------------------------------------------------------------
#   Boundary conditions
# --------------------------------------------------------------------------

@cuda.jit(device=True)
def relative_2d(x0, y0, a0, x1, y1, a1, rmax, arena, arena_X, arena_Y, periodic_X, periodic_Y):
  '''
  Relative position and orientation between two agents
  The output is tuple (z, alpha, status) containing the relative complex polar
  coordinates z, the relative orientation alpha and the visibility status.
  NB: in case the distance is above rmax, (0,0,False) is returned
  '''
  
  if arena==Arena.CIRCULAR.value:
    '''
    Circular arena
    '''

    # Complex polar coordinates
    z = complex(x1-x0, y1-y0)

  elif arena==Arena.RECTANGULAR.value:
    '''
    Rectangular arena
    '''

    # dx
    if periodic_X:
      dx = x1-x0 if abs(x1-x0)<=arena_X else ( x1-x0-2*arena_X if x1>x0 else x1-x0+2*arena_X )
    else:
      dx = x1-x0

    # dy
    if periodic_Y:
      dy = y1-y0 if abs(y1-y0)<=arena_Y else ( y1-y0-2*arena_Y if y1>y0 else y1-y0+2*arena_Y )
    else:
      dy = y1-y0

    # Complex polar coordinates
    z = complex(dx, dy)

  # Out of sight agents
  if rmax is not None and abs(z)>rmax: return (0, 0, False)

  return (z, a1-a0, True)

@cuda.jit(device=True)
def assign_2d(z0, z1, v, a, arena, arena_X, arena_Y, periodic_X, periodic_Y):
  
  if v==0:
    return (z1.real, z1.imag, v, a)

  if arena==Arena.CIRCULAR.value:
    '''
    Circular arena
    '''

    # Check for outsiders
    if abs(z1) > arena_X:
      '''
      Reflexive circular
      (Periodic boundary conditions are not possible with a circular arena)
      '''

      # Crossing point
      phi = a + math.asin((z0.imag*math.cos(a) - z0.real*math.sin(a))/arena_X)
      zc = cmath.rect(arena_X, phi)

      # Position        
      z1 = zc + (v-abs(zc-z0))*cmath.exp(1j*(cmath.pi + 2*phi - a))

      # Final velocity
      a += cmath.pi-2*(a-phi)

    # Final position
    px = z1.real
    py = z1.imag


  elif arena==Arena.RECTANGULAR.value:
    '''
    Rectangular arena
    '''

    zv = cmath.rect(v, a)
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

    v, a = cmath.polar(complex(vx, vy))

  return (px, py, v, a)
