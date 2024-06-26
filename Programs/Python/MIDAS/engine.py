'''
MIDAS Engine
'''

import os
import warnings
import math, cmath
import time
import numpy as np
import numba as nb
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32

from Animation.Window import Window

from MIDAS.storage import Storage
from MIDAS.enums import *
import MIDAS.animation
import MIDAS.verbose

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

  def get_param(self):

    param = []

    # --- Parameter serialization ------------------------------------------

    match self.dimension:

      case 1:

        param = np.zeros(3, dtype=np.float32)

        # Arena shape
        param[0] = self.arena.value

        # Arena size
        param[1] = self.arena_shape[0]/2

        # Arena periodicity
        param[2] = self.periodic if self.arena==Arena.CIRCULAR else self.periodic[0]

      case 2:

        param = np.zeros(5, dtype=np.float32)

        # Arena shape
        param[0] = self.arena.value

        # Arena size
        param[1] = self.arena_shape[0]/2
        param[2] = self.arena_shape[1]/2

        # Arena periodicity
        param[3] = self.periodic if self.arena==Arena.CIRCULAR else self.periodic[0]
        param[4] = self.periodic if self.arena==Arena.CIRCULAR else self.periodic[1]

      case 3:

        param = np.zeros(7, dtype=np.float32)

        # Arena shape
        param[0] = self.arena.value

        # Arena size
        param[1] = self.arena_shape[0]/2
        param[2] = self.arena_shape[1]/2
        param[3] = self.arena_shape[2]/2

        # Arena periodicity
        param[4] = self.periodic if self.arena==Arena.CIRCULAR else self.periodic[0]
        param[5] = self.periodic if self.arena==Arena.CIRCULAR else self.periodic[1]
        param[6] = self.periodic if self.arena==Arena.CIRCULAR else self.periodic[2]

    return param

# === AGENTS ===============================================================

class Agents:
  '''
  Collection of all agents
  '''

  def __init__(self, dimension):

    # Dimension
    self.dimension = dimension

    # Number of agents
    self.N = 0

    # Types
    # self.atype = np.empty(0)

    # Positions and velocities
    '''
    Position are expressed in cartesian coordinates (x,y,z)
    Velocities are expressed in polar coordinates (v,alpha,beta)
    '''
    self.pos = np.empty((0, dimension))
    self.vel = np.empty((0, dimension))

    # Group
    self.group = np.empty(0)

    # Limits
    self.vmin = np.empty(0)
    self.vmax = np.empty(0)
    self.rmax = np.empty(0)
    if self.dimension>1: self.damax = np.empty(0)
    if self.dimension>2: self.dbmax = np.empty(0)

    # Noise
    self.vnoise = np.empty(0)
    if self.dimension>1: self.danoise = np.empty(0)
    if self.dimension>2: self.dbnoise = np.empty(0)    

  def get_param(self):

    tmp = [self.group, self.vmin, self.vmax, self.rmax]
    if self.dimension>1: tmp.append(self.damax)
    if self.dimension>2: tmp.append(self.dbmax)
    tmp.append(self.vnoise)
    if self.dimension>1: tmp.append(self.danoise)
    if self.dimension>2: tmp.append(self.dbnoise)

    return np.column_stack(tmp)

# === GROUPS ===============================================================

class Groups:
  '''
  Subcollection of agents
  
  NB: All agents in a group have the same type
  '''

  def __init__(self, dimension=None):
  
    self.dimension = dimension

    # Number of groups
    self.N = 0

    # Group names
    self.names = []

    # Types of agents
    self.atype = []

    # I/O
    self.inputs = []
    self.outputs = []

  # ------------------------------------------------------------------------
  #   Parameter serialization
  # ------------------------------------------------------------------------

  def get_param(self, inputs):
    '''
    Groups parameters
    '''
   
    l_gparam = []

    for gid, atype in enumerate(self.atype):

      # Przpare row
      row = [atype.value]

      match atype:

        case Agent.FIXED:

          # === FIXED AGENTS =================================================

          pass

        case Agent.RIPO:

          In = self.inputs[gid]
          Out = self.outputs[gid]

          # Numbers
          row.append(len(In))
          row.append(np.sum([inputs[i].coefficients.size for i in In]))
          row.append(len(Out))

          # Lists
          [row.append(x) for x in In]
          [row.append(x) for x in Out]

      l_gparam.append(np.array(row))
    
    return l_gparam

# === GRIDS ===============================================================

class PolarGrid:
  '''
  Polar grid
  '''

  def __init__(self, rZ=[], rmax=-1, nSa=1, nSb=1):

    self.rZ = np.array(rZ)
    self.nR = self.rZ.size + 1
    self.rmax = rmax

    self.nSa = nSa
    self.nSb = nSb    

    self.nZ = self.nR*self.nSa*self.nSb
    
# === INPUTS ===============================================================

class Input:
  '''
  Input
  '''

  def __init__(self, perception, **kwargs):

    # Perception type
    self.perception = perception

    # Normalization
    self.normalization = kwargs['normalization'] if 'normalization' in kwargs else Normalization.NONE

    # Grid
    self.grid = kwargs['grid'] if 'grid' in kwargs else None
    self.coefficients = np.array(kwargs['coefficients']) if 'coefficients' in kwargs else None

  def get_param(self, dimension, nG, **kwargs):
    '''
    Perception parameters
    '''
   
    # Initialization
    param = [self.perception.value, self.normalization.value]

    # Grid parameters
    param.append(self.grid.nR)
    param.append(self.grid.rmax)
    if dimension>1: param.append(self.grid.nSa)
    if dimension>2: param.append(self.grid.nSb)
    param.append(self.grid.nZ)

    # Coefficients
    param.append(self.coefficients.size)
    [param.append(c) for c in self.coefficients]

    return np.array(param)

# === OUPUTS ===============================================================

class Output:
  '''
  Output
  '''

  def __init__(self, action, **kwargs):

    self.action = action
    self.activation = kwargs['activation'] if 'activation' in kwargs else Activation.IDENTITY

  def get_param(self, **kwargs):
    '''
    Output parameters
    '''
   
    return [self.action.value, self.activation.value]

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
    self.inputs = []
    self.outputs = []

    # Parameters
    self.param_geometry = None
    self.param_agents = None
    self.param_perceptions = None
    self.param_outputs = None
    self.param_groups = None
    
    # Storage
    self.storage = None

    # Animation
    self.window = None
    self.animation = None

    # GPU
    self.cuda = None
    
    # --- Time

    # Total number of steps
    self.steps = None

    # Computation time reference
    self.tref = None

    # --- Misc attributes
    
    self.verbose = MIDAS.verbose.cli_Reporter()
    self.verbose.level = Verbose.NORMAL

# ------------------------------------------------------------------------
  #   Additions
  # ------------------------------------------------------------------------

  def add_input(self, perception, **kwargs):

    self.inputs.append(Input(perception=perception, **kwargs))
    return len(self.inputs)-1

  def add_output(self, action, **kwargs):

    self.outputs.append(Output(action, **kwargs))
    return len(self.outputs)-1

  def add_group(self, gtype, N, **kwargs):

    # Group name
    gname = kwargs['name'] if 'name' in kwargs else gtype.name

    # --- Initial conditions -----------------------------------------------

    # --- User definition

    if 'initial_condition' in kwargs:
      initial_condition = kwargs['initial_condition']
    else:
      initial_condition = {'position': None, 'orientation': None, 'speed': Default.vmax.value}

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
    
    # --- Agents definition ------------------------------------------------

    self.agents.N += N

    # Position and speed
    self.agents.pos = np.concatenate((self.agents.pos, pos), axis=0)
    self.agents.vel = np.concatenate((self.agents.vel, vel), axis=0)

    # --- Groups definition ------------------------------------------------
    
    # Groups
    if gname in self.groups.names:
      iname = self.groups.names.index(gname)
    else:
      iname = len(self.groups.names)
      self.groups.N += 1
      self.groups.names.append(gname)
      self.groups.atype.append(gtype)

    # --- Agents specifications --------------------------------------------

    def arrify(v):
      if isinstance(v, list): v = np.array(v)
      if not isinstance(v, np.ndarray): v = np.full(N, v)
      return v

    # Agents' groups
    group = arrify(iname)

    # Speed limits
    vmin = arrify(kwargs['vmin'] if 'vmin' in kwargs else Default.vmin.value)
    vmax = arrify(kwargs['vmax'] if 'vmax' in kwargs else V)
    
    # Visibility limit
    rmax = arrify(kwargs['rmax'] if 'rmax' in kwargs else Default.rmax.value)

    # Reorientation limits
    if self.geom.dimension>1: damax = arrify(kwargs['damax'] if 'damax' in kwargs else Default.damax.value)
    if self.geom.dimension>2: dbmax = arrify(kwargs['dbmax'] if 'dbmax' in kwargs else Default.dbmax.value)

    # Noise
    if 'noise' in kwargs:
      vnoise = arrify(kwargs['noise'][0])
      if self.geom.dimension>1: danoise = arrify(kwargs['noise'][1]) 
      if self.geom.dimension>2: dbnoise = arrify(kwargs['noise'][2]) 
    else:
      vnoise = arrify(kwargs['vnoise'] if 'vnoise' in kwargs else Default.vnoise.value)
      if self.geom.dimension>1: danoise = arrify(kwargs['danoise'] if 'danoise' in kwargs else Default.danoise.value)
      if self.geom.dimension>2: dbnoise = arrify(kwargs['dbnoise'] if 'dbnoise' in kwargs else Default.dbnoise.value)

    # --- Concatenations

    self.agents.group = np.concatenate((self.agents.group, group), axis=0)
    self.agents.vmin = np.concatenate((self.agents.vmin, vmin), axis=0)
    self.agents.vmax = np.concatenate((self.agents.vmax, vmax), axis=0)
    self.agents.rmax = np.concatenate((self.agents.rmax, rmax), axis=0)
    self.agents.vnoise = np.concatenate((self.agents.vnoise, vnoise), axis=0)
    if self.geom.dimension>1:
      self.agents.damax = np.concatenate((self.agents.damax, damax), axis=0)
      self.agents.danoise = np.concatenate((self.agents.danoise, danoise), axis=0)
    if self.geom.dimension>2:
      self.agents.dbmax = np.concatenate((self.agents.dbmax, dbmax), axis=0)
      self.agents.dbnoise = np.concatenate((self.agents.dbnoise, dbnoise), axis=0)

    # --- Groups specifications --------------------------------------------

    self.groups.inputs.append(kwargs['inputs'] if 'inputs' in kwargs else [])
    self.groups.outputs.append(kwargs['outputs'] if 'outputs' in kwargs else [])

  # ------------------------------------------------------------------------
  #   Setups
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

  def setup_storage(self, db_file):
    '''
    Setup the sqlite database for storage

    NB: DB initialization (creation and initial filling) is performed during 
    self.run().
    '''

    self.storage = Storage(db_file, verbose=self.verbose)

  # ------------------------------------------------------------------------
  #   Run process
  # ------------------------------------------------------------------------

  def define_parameters(self):
    '''
    Define parameter arrays
    '''

    # --- Geometry ---------------------------------------------------------

    self.param_geometry = self.geom.get_param()

    # --- Agent parameters -------------------------------------------------  

    self.param_agents = self.agents.get_param()

    # --- Input parameters -------------------------------------------------

    l_pparam = [I.get_param(self.geom.dimension, self.groups.N) for I in self.inputs]

    # Adjust columns size
    ncol = max([row.size for row in l_pparam])
    for i, row in enumerate(l_pparam):
      l_pparam[i] = np.pad(row, (0, ncol-row.size))

    # Concatenate
    self.param_perceptions = np.row_stack(l_pparam) 

    # --- Output parameters ------------------------------------------------

    l_oparam = [Out.get_param() for Out in self.outputs]

    # Concatenate
    self.param_outputs = np.row_stack(l_oparam) 

    # --- Group parameters -------------------------------------------------

    l_gparam = self.groups.get_param(self.inputs)
    
    # Adjust columns size
    ncol = max([row.size for row in l_gparam])
    for i, row in enumerate(l_gparam):
      l_gparam[i] = np.pad(row, (0, ncol-row.size))

    # Concatenate
    self.param_groups = np.row_stack(l_gparam)

  def run(self):

    # === Checks ===========================================================

    # No animation
    if self.animation is None:
    
      # Number of steps
      if self.steps is None:
        warnings.warn('The number of steps must be defined when there is no visualization.')
        return
      
      # Storage
      if self.storage is None:
        warnings.warn('A storage location must be defined when there is no visualization.')
        return

    # === Preparation ======================================================

    if self.storage is not None:

      # Initialize storage
      self.storage.initialize(self)

      # Initial state
      self.storage.insert_step(0, self.agents.pos, self.agents.vel)

    if self.verbose.level>=Verbose.NORMAL:

      self.verbose.line()
      self.verbose(f'Starting simulation with {self.agents.N} agents over {self.steps} steps')

    # Reference time
    self.tref = time.time()

    # --- CUDA preparation -------------------------------------------------

    # Define parameters (for CUDA)
    self.define_parameters()

    # CUDA object
    self.cuda = CUDA(self)

    # --- Send arrays to device

    # Parameters
    self.cuda.geometry = cuda.to_device(self.param_geometry.astype(np.float32))
    self.cuda.agents = cuda.to_device(self.param_agents.astype(np.float32))
    self.cuda.perceptions = cuda.to_device(self.param_perceptions.astype(np.float32))
    self.cuda.actions = cuda.to_device(self.param_outputs.astype(np.float32))
    self.cuda.groups = cuda.to_device(self.param_groups.astype(np.float32))

    # Double buffers
    self.cuda.p0 = cuda.to_device(self.agents.pos.astype(np.float32))
    self.cuda.v0 = cuda.to_device(self.agents.vel.astype(np.float32))
    self.cuda.p1 = cuda.device_array((self.agents.N, self.geom.dimension), np.float32)
    self.cuda.v1 = cuda.device_array((self.agents.N, self.geom.dimension), np.float32)

    # --- Main loop --------------------------------------------------------

    if self.animation is None:

      from alive_progress import alive_bar

      '''
      It is important that steps start at 1, step=0 being the initial state
      '''

      with alive_bar(self.steps) as bar:
        
        bar.title = self.verbose.get_caller(1)
        for step in range(self.steps):
          if step: self.step(step)
          bar()

      self.end()

    else:

      # Use the animation clock
      self.animation.initialize()
      self.window.show()

  def step(self, i):

    # Double-buffer computation trick
    if i % 2:
      
      self.cuda.step[self.cuda.gridDim, self.cuda.blockDim](self.cuda.geom,
        self.cuda.p0, self.cuda.v0, self.cuda.p1, self.cuda.v1,
        self.cuda.aparam, self.cuda.gparam, self.cuda.rng)
      
      cuda.synchronize()
      
      self.agents.pos = self.cuda.p1.copy_to_host()
      self.agents.vel = self.cuda.v1.copy_to_host()

    else:

      self.cuda.step[self.cuda.gridDim, self.cuda.blockDim](self.cuda.geom,
        self.cuda.p1, self.cuda.v1, self.cuda.p0, self.cuda.v0,
        self.cuda.aparam, self.cuda.gparam, self.cuda.rng)
      
      cuda.synchronize()
      
      self.agents.pos = self.cuda.p0.copy_to_host()
      self.agents.vel = self.cuda.v0.copy_to_host()
    
    # --- DB Storage

    if self.storage is not None:
      self.storage.insert_step(i, self.agents.pos, self.agents.vel)

    # --- End of simulation (animation)

    if self.animation is not None and self.steps is not None and i>=self.steps-1:
      self.end()

  def end(self):
    '''
    Operations to do when the simalutation is over
    '''
    # End of simulation
    self.verbose('End of simulation @ {:d} steps ({:.2f} s)'.format(self.window.step, time.time()-self.tref))
    self.verbose.line()

    # End storage
    if self.storage is not None:
      self.storage.db_conn.commit()

    # End display
    if self.animation is not None:
      self.animation.is_running = False
      self.animation.window.close()

############################################################################
############################################################################
# #                                                                      # #
# #                                                                      # #
# #                               CUDA                                   # #
# #                                                                      # #
# #                                                                      # #
############################################################################
############################################################################

'''

Vocabulary:
  - A grid is a polar grid, defined for each perception function
  - A grid is composed of zones
  - A set of inputs in a griven grid is called a 'perception'

=== PARAMETERS =============================================================

On the cuda side the general model is decomposed in different parameter sets:

          [Agent parameters]  (N rows)
aparam/agents
  ├── group           (1)     group index
  ├── vlim            (2)     speed limits (vmin, vmax)
  ├── rmax            (1)     maximal distance for visibility
  ├── damax           (dim-1) reorientation limits (damax, dbmax)
  └── noise           (dim)   (vnoise, anoise, bnoise)

           [Input parameters] (nP rows)
pparam/perceptions
  ├── ptype           (1)     perception type
  ├── ntype           (1)     normalization type
  ├── nR              (1)     ┐
  ├── rmax            (1)     │
  ├── nSa  [if dim>1] (1)     │ Grid definition
  ├── nSb  [if dim>2] (1)     │
  ├── rZ              (nR-1)  ┘
  ├── nC              (1)     number of coefficients
  └── weights         (var)   ┐
      ├── w0          (1)     │ as many as weights
      └── ...                 ┘

          [Output parameters] (nP rows)
oparam/actions
  ├── otype           (1)     output type
  └── ftype           (1)     activation type

          [Group parameters]  (nG rows)
gparam/groups
  ├── atype           (1)     type of the agents in the group  
  ├── nP              (1)     number of perceptions
  ├── nI              (1)     number of inputs (= number of weights)
  ├── nO              (1)     number of outputs
  ├── perceptions     (var)   ┐
  │   ├── p0          (1)     │ as many as perceptions
  │   └── ...                 ┘
  └── outputs         (var)   ┐
      ├── o0          (1)     │ as many as outputs
      └── ...                 ┘

=== DEVICE LOCAL ARRAYS ==================================================

The following local arrays are defined:

* Other agents:
  - z         (N, nb.complex64)   relative position
  - alpha     (N, nb.float32)     relative orientation
  - visible   (N, nb.boolean)     visibility (= is distance below rmax)

* RIPO
  - rs        (m_nR, nb.float32)  zones radii
  - input     (m_nI, nb.float32)  inputs
  - weights   (m_nI, nb.float32)  weights

=== PERCEPTION DEFINITION ==================================================

In the perception file, there should be:
- a definition for the IntEnum  'Perception', containing PRESENCE and ORIENTATION.
- a device function managing all the cases.
'''

class CUDA:

  def __init__(self, engine):

    # Associated engine
    self.engine = engine

    # Blocks and grid size
    self.blockDim = 32
    self.gridDim = (engine.agents.N + (self.blockDim - 1)) // self.blockDim

    # Double buffers
    self.p0 = None
    self.v0 = None
    self.p1 = None
    self.v1 = None

    # Parameter arrays
    self.geometry = None
    self.agents = None
    self.perceptions = None
    self.actions = None
    self.groups = None
    
    # Random number generator
    self.rng = create_xoroshiro128p_states(self.blockDim*self.gridDim, seed=0)

    # --------------------------------------------------------------------------
    #   CUDA kernel variables
    # --------------------------------------------------------------------------

    # CUDA local array dimensions
    # N = self.engine.agents.N
    # m_nR = max(self.engine.groups.l_nR)
    # m_nCaf = max(self.engine.groups.l_nCaf)
    # m_nCad = m_nCaf*self.engine.groups.N
    # m_nI = max(self.engine.groups.l_nI)

    # print('gparam', self.engine.groups.param)
    # print('m_nR', m_nR)
    # print('m_nZ', m_nZ)
    # print('m_nCaf', m_nCaf)
    # print('m_nCad', m_nCad)
    # print('m_nI', m_nI)

    
    # from test_package import test_fun
    # test = test_fun

    # Z = np.array([test_fun],dtype=object)
    # test = 18

    # --------------------------------------------------------------------------
    #   The CUDA kernel
    # --------------------------------------------------------------------------
    
    @cuda.jit
    def CUDA_step(geom, p0, v0, p1, v1, aparam, gparam, rng):
      '''
      The CUDA kernel
      '''

      i = cuda.grid(1)

      if i<p0.shape[0]:

        # if i==0:
        #   test_fun()
          
        # Dimension
        dim = p0.shape[1]

        # Group id
        gid = int(aparam[i, 0])

        # Agent type
        atype = int(gparam[gid, 0])
        
        # === Fixed points =====================================================

        if atype==Agent.FIXED.value:
          for j in range(dim):
            p1[i,j] = p0[i,j]

        # === Extracting parameters ============================================

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
        
        # Velocity limits
        vmin = aparam[i,1]
        vmax = aparam[i,2]

        # Visibility limit
        rmax = aparam[i,3]

        match dim:

          case 1:
            vnoise = aparam[i,4]

          case 2:
            damax = aparam[i,4]
            vnoise = aparam[i,5]
            anoise = aparam[i,6]

          case 3:
            damax = aparam[i,4]
            dbmax = aparam[i,5]
            vnoise = aparam[i,6]
            anoise = aparam[i,7]
            bnoise = aparam[i,8]

        # === Computation ======================================================

        # Agent polar coordinates
        v = v0[i,0]
        a = v0[i,1]

        # Other agents relative coordinates
        z = cuda.local.array(N, nb.complex64)
        alpha = cuda.local.array(N, nb.float32)
        visible = cuda.local.array(N, nb.boolean)

        for j in range(N):

          # Skip self-perception
          if j==i: continue

          # Distance and relative orientation
          match dim:
            case 1: pass
            case 2:
              z[j], alpha[j], visible[j] = relative_2d(p0[i,0], p0[i,1], v0[i,1], p0[j,0], p0[j,1], v0[j,1], rmax, arena, arena_X, arena_Y, periodic_X, periodic_Y)
            case 3: pass

        # --- RIPO agents ------------------------------------------------------

        # if atype==Agent.RIPO.value:

        #   # Number of groups
        #   nG = gparam.shape[0]

        #   # === Deserialization of the RIPO parameters ===
          

        #   # --- Radial limits

        #   # Number of zones per slice
        #   nR = int(gparam[gid, 1])

        #   # --- Angular slices

        #   nSa = int(gparam[gid, 2]) if dim>1 else 1
        #   nSb = int(gparam[gid, 3]) if dim>2 else 1

        #   # --- Zones radii

        #   rS = cuda.local.array(m_nR, nb.float32)
        #   for ri in range(nR-1):
        #     rS[ri] = gparam[gid, dim+ri+1]

        #   # Maximal radius
        #   rmax = gparam[gid, nR + dim] if gparam[gid, nR + dim]>0 else None

        #   # --- Inputs / outputs

        #   # Number of input sets
        #   nIs = int(gparam[gid, nR + dim + 1])

        #   # Number of outputs
        #   nOut = int(gparam[gid, nR + dim + 2])

        #   # Input index reference
        #   kIref = nR + dim + 3

        #   # Number of coefficients per input type
        #   nc_AFI = nOut*nR*nSa*nSb
        #   nc_ADI = nOut*nG*nR*nSa*nSb

        #   # --- Weights

        #   weights = cuda.local.array(m_nI, nb.float32)
                
        #   # Default inputs
        #   i_pres = None
        #   i_ornt = None
        #   i_orntC = None

        #   # Default mode
        #   bADInput = False

        #   # Scan inputs
        #   k = kIref
        #   nIn = 0
        #   for iS in range(nIs):

        #     match gparam[gid, k]:                  

        #       case Perception.PRESENCE.value:
        #         bADInput = True
        #         i_pres = cuda.local.array(m_nCad, dtype=nb.float32)

        #         # Store weights
        #         for ci in range(nc_ADI):
        #           weights[nIn] = gparam[gid, k + 2 + ci]
        #           nIn += 1

        #         # Update input index
        #         k += nc_ADI + 2

        #       case Perception.ORIENTATION.value:
        #         bADInput = True
        #         i_ornt = cuda.local.array(m_nCad, dtype=nb.float32)
        #         i_orntC = cuda.local.array(m_nCad, dtype=nb.complex64)

        #         # Store weights
        #         for ci in range(nc_ADI):
        #           weights[nIn] = gparam[gid, k + 2 + ci]
        #           nIn += 1

        #         # Update input index
        #         k += nc_ADI + 2

        #   # --- Outputs

        #   # Output index reference
        #   kOref = k

        #   Out_da = -1
        #   Out_dv = -1

        #   for io in range(nOut):

        #     match gparam[gid, kOref+io*2]:

        #       case Output.REORIENTATION.value:
        #         Out_da = gparam[gid, kOref+io*2+1]

        #       case Output.SPEED_MODULATION.value:
        #         Out_dv = gparam[gid, kOref+io*2+1]

        #   # === Agent-free perception ======================================

        #   # TO DO

        #   # === Agent-dependent perception =================================

        #   if bADInput:

        #     for j in range(N):

        #       # Skip self-perception
        #       if i==j: continue

        #       # Distance and relative orientation
        #       z, alpha, status = relative_2d(p0[i,0], p0[i,1], v0[i,1], p0[j,0], p0[j,1], v0[j,1], rmax, arena, arena_X, arena_Y, periodic_X, periodic_Y)

        #       # Skip agents out of reach (further than rmax)
        #       if not status: continue

        #       # --- Index in the grid

        #       # Radial index
        #       ri = 0
        #       for k in range(nR):
        #         ri = k
        #         if abs(z)<rS[k]: break
                
        #       ai = int((cmath.phase(z) % (2*math.pi))/2/math.pi*nSa) if dim>1 else 0
        #       bi = 0 # if dim>2 else 0  # TODO: 3D
                                
        #       match dim:
        #         case 1: ig = ri
        #         case 2: ig = ri*nSa + ai
        #         case 3: ig = (ri*nSa + ai)*nSb + bi

        #       # --- Inputs

        #       if i_pres is not None:
        #         i_pres[ig] += 1

        #       if i_ornt is not None:
        #         # TODO: Implement other dimensions
        #         match dim:
        #           case 1: pass
        #           case 2:
        #             i_orntC[ig] += cmath.rect(1., alpha)
        #           case 3: pass

        #   # Orientation
        #   if i_ornt is not None:
        #     for zi in range(nc_ADI):
        #       i_ornt[zi] = cmath.phase(i_orntC[zi])

        #   # === Inputs and normalizaton ====================================

        #   # --- Normalization
        
        #   # Weighted sum
        #   WS = 0
          
        #   k = kIref
        #   for iS in range(nIs):

        #     match gparam[gid, k]:                  

        #       case Perception.PRESENCE.value:

        #         match gparam[gid, k+1]:

        #           case Normalization.NONE.value:

        #             for zi in range(nc_ADI):
        #               WS += i_pres[zi]*weights[zi]

        #           case Normalization.SAME_RADIUS.value: pass
        #           case Normalization.SAME_SLICE.value: pass
        #           case Normalization.ALL.value: pass

        #         # Update k
        #         k += nc_ADI + 2

        #       case Perception.ORIENTATION.value:
                
        #         match gparam[gid, k+1]:

        #           case Normalization.NONE.value:

        #             for zi in range(nc_ADI):
        #               WS += i_ornt[zi]*weights[zi]

        #           case Normalization.SAME_RADIUS.value: pass
        #           case Normalization.SAME_SLICE.value: pass
        #           case Normalization.ALL.value: pass

        #         k += nc_ADI + 2

          # # === Processing =================================================

          # # --- Reorientation

          # match Out_da:

          #   case Activation.ANGLE.value:
          #     da = damax*(4/math.pi*math.atan(math.exp((WS)/2))-1)

          #   case _:
          #     da = 0

          # # --- Speed modulation
          
          # match Out_dv:

          #   case Activation.SPEED.value:
          #     dv = 0

          #   case _:
          #     dv = 0
      
        da = 0
        dv = 0

        # === Update =======================================================

        match dim:

          case 2:

            # Update velocity
            v += dv
            a += da

            # --- Noise ----------------------------------------------------

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

    # Store CUDA kernel
    self.step = CUDA_step

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

  # Orientation
  z *= cmath.rect(1., -a0)

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
