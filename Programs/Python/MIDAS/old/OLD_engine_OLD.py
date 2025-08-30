'''
MIDAS Engine
'''

# import importlib
# import warnings
# import math, cmath
# import time
# import numpy as np
# import numba as nb
# from numba import cuda
# from numba.cuda.random import create_xoroshiro128p_states

# from Animation.Window import Window

import MIDAS

# from MIDAS.enums import *
# from MIDAS.polar_grid import PolarGrid
# from MIDAS.fields import Fields
# from MIDAS.coefficients import Coefficients
# import MIDAS.network
# from MIDAS.storage import Storage
# import MIDAS.engine.animation
# from MIDAS.information import InformationBase
# import MIDAS.verbose

# === GEOMETRY =============================================================

class Geometry:
  '''
  Geometry of the simulation, including:
  - Dimension
  - Arena type and shape
  - Boundary conditions
  '''

  # ========================================================================
  def __init__(self, dimension, **kwargs):

    # --- Dimension

    self.dimension = dimension

    # --- Arena

    # Arena shape ('circular' or 'rectangular')
    self.arena = kwargs['arena'] if 'arena' in kwargs else Arena.RECTANGULAR
    self.arena_shape =  np.array(kwargs['shape'] if 'shape' in kwargs else [1]*self.dimension)

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
  
  # ========================================================================
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
  
  def set_initial_orientations(self, orientation, n):
    '''
    Initial velocities
    '''
      
    if orientation in [None, 'random', 'shuffle']:
      orientation = 2*np.pi*np.random.rand(n)

    return orientation

  def get_param(self):

    match self.dimension:

      case 1:

        param = np.zeros(4, dtype=np.float32)

        # Dimension
        param[0] = self.dimension

        # Arena shape
        param[1] = self.arena.value

        # Arena size
        param[2] = self.arena_shape[0]/2

        # Arena periodicity
        param[3] = self.periodic if self.arena==Arena.CIRCULAR else self.periodic[0]

      case 2:

        param = np.zeros(6, dtype=np.float32)

        # Dimension
        param[0] = self.dimension

        # Arena shape
        param[1] = self.arena.value

        # Arena size
        param[2] = self.arena_shape[0]/2
        param[3] = self.arena_shape[1]/2

        # Arena periodicity
        param[4] = self.periodic if self.arena==Arena.CIRCULAR else self.periodic[0]
        param[5] = self.periodic if self.arena==Arena.CIRCULAR else self.periodic[1]

      case 3:

        param = np.zeros(8, dtype=np.float32)

        # Dimension
        param[0] = self.dimension

        # Arena shape
        param[1] = self.arena.value

        # Arena size
        param[2] = self.arena_shape[0]/2
        param[3] = self.arena_shape[1]/2
        param[4] = self.arena_shape[2]/2

        # Arena periodicity
        param[5] = self.periodic if self.arena==Arena.CIRCULAR else self.periodic[0]
        param[6] = self.periodic if self.arena==Arena.CIRCULAR else self.periodic[1]
        param[7] = self.periodic if self.arena==Arena.CIRCULAR else self.periodic[2]

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

    # Scales
    self.dv_scale = np.empty(0)
    if self.dimension>1: self.da_scale = np.empty(0)
    if self.dimension>2: 
      self.db_scale = np.empty(0)
      self.dc_scale = np.empty(0)

    # Noise
    self.vnoise = np.empty(0)
    if self.dimension>1: self.anoise = np.empty(0)
    if self.dimension>2: 
      self.bnoise = np.empty(0)
      self.cnoise = np.empty(0)

  def get_param(self):

    tmp = [self.group, self.vmin, self.vmax, self.rmax, self.dv_scale]
    if self.dimension>1: tmp.append(self.da_scale)
    if self.dimension>2: 
      tmp.append(self.db_scale)
      tmp.append(self.dc_scale)
    tmp.append(self.vnoise)
    if self.dimension>1: tmp.append(self.anoise)
    if self.dimension>2:
      tmp.append(self.bnoise)
      tmp.append(self.cnoise)

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

      # Prepare row
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
          row.append(len(Out))

          # Lists
          [row.append(x) for x in In]
          [row.append(x) for x in Out]

      l_gparam.append(np.array(row))
    
    return l_gparam

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

    # Field
    self.field = kwargs['field'] if 'field' in kwargs else 0

    # Weights (set after groups definitions)
    self._coefficients =  None
    self.weights = None

  def get_param(self, dimension, nG, field_offset, **kwargs):
    '''
    Perception parameters
    '''
   
    # Initialization
    param = [self.perception.value, self.normalization.value]

    # Field offset
    param.append(field_offset)   

    # Grid parameters
    if self.grid is None:
      param.append(1)                 # nR
      param.append(-1)                # rmax
      if dimension>1: param.append(1) # nSa
      if dimension>2: param.append(1) # nSb

    else:
      param.append(self.grid.nR)
      param.append(self.grid.rmax)
      if dimension>1: param.append(self.grid.nSa)
      if dimension>2: param.append(self.grid.nSb)
      [param.append(x) for x in self.grid.rZ]

    # weights
    param.append(self.weights.size)

    match dimension:
      case 1: pass
      case 2:
        [param.append(w) for w in self.weights]
      case 3: pass

    return np.array(param)

# --- Coefficients ---------------------------------------------------------

  @property
  def coefficients(self): return self._coefficients

  @coefficients.setter
  def coefficients(self, C):
    self._coefficients = C
    self.weights = C.to_weights()
    
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

class engine:
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
    self.fields = Fields(self)
    self.inputs = []
    self.outputs = []
    
    # Storage
    self.storage = None

    # Animation
    self.window = None
    self.animation = None
    self.information = None

    # --- GPU

    self.cuda = None        

    # Parameters for the kernel
    self.param_geometry = None
    self.param_agents = None
    self.param_perceptions = None
    self.param_outputs = None
    self.param_groups = None
    self.param_custom = None

    # CUDA variables
    self.agent_drivenity = False

    # --- Customizable CUDA packages

    self.CUDA_perception = None
    self.CUDA_action = None

    self.n_CUDA_properties = 0
    self.properties = None

    # --- Time

    # Total number of steps
    self.steps = None

    # Computation time reference
    self.tref = None

    # --- Misc attributes
    
    self.custom_param = {}
    self.verbose = MIDAS.verbose.cli_Reporter()
    self.verbose.level = Verbose.NORMAL

# ------------------------------------------------------------------------
  #   Additions
  # ------------------------------------------------------------------------

  def add_input(self, perception, **kwargs):

    # Field grids
    match perception:
      case Perception.FIELD:
        nSa = kwargs['nSa'] if 'nSa' in kwargs else 4
        kwargs['grid'] = PolarGrid(nSa=nSa)

    # Append input
    self.inputs.append(Input(perception, **kwargs))

    # Check agent-drivenity
    if ('agent_drivenity' in kwargs and kwargs['agent_drivenity']) \
      or perception in [Perception.PRESENCE, Perception.ORIENTATION]:

      self.agent_drivenity = True

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
      if isinstance(initial_condition['position'][0], str):
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

    match gtype:

      case Agent.FIXED:

        tmp = arrify(0)

        vmin = tmp
        vmax = tmp
        rmax = tmp
        dv_scale = tmp
        vnoise = tmp
        if self.geom.dimension>1: 
          da_scale = tmp
          anoise = tmp
        if self.geom.dimension>2: 
          db_scale = tmp
          dc_scale = tmp
          bnoise = tmp
          cnoise = tmp

      case _:

        # Speed limits
        vmin = arrify(kwargs['vmin'] if 'vmin' in kwargs else Default.vmin.value)
        vmax = arrify(kwargs['vmax'] if 'vmax' in kwargs else V)
        
        # Visibility limit
        if 'rmax' in kwargs and kwargs['rmax'] is not None:
          rmax = arrify(kwargs['rmax'])
        else:
          l_rmax = [I.grid.rmax if I.grid is not None else -1 for I in self.inputs]
          rmax = arrify(-1 if any([x==-1 for x in l_rmax]) else max(l_rmax))

        # Reorientation scales
        dv_scale = arrify(kwargs['dv_scale'] if 'dv_scale' in kwargs else Default.dv_scale.value)
        if self.geom.dimension>1: 
          da_scale = arrify(kwargs['da_scale'] if 'da_scale' in kwargs else Default.da_scale.value)
        if self.geom.dimension>2: 
          db_scale = arrify(kwargs['db_scale'] if 'db_scale' in kwargs else Default.db_scale.value)
          dc_scale = arrify(kwargs['dc_scale'] if 'dc_scale' in kwargs else Default.dc_scale.value)

        # Noise
        if 'noise' in kwargs:
          vnoise = arrify(kwargs['noise'][0])
          if self.geom.dimension>1: anoise = arrify(kwargs['noise'][1]) 
          if self.geom.dimension>2: 
            bnoise = arrify(kwargs['noise'][2]) 
            cnoise = arrify(kwargs['noise'][3])
        else:
          vnoise = arrify(kwargs['vnoise'] if 'vnoise' in kwargs else Default.vnoise.value)
          if self.geom.dimension>1:
            anoise = arrify(kwargs['anoise'] if 'anoise' in kwargs else Default.anoise.value)
          if self.geom.dimension>2: 
            bnoise = arrify(kwargs['bnoise'] if 'bnoise' in kwargs else Default.bnoise.value)
            cnoise = arrify(kwargs['cnoise'] if 'cnoise' in kwargs else Default.cnoise.value)

    # --- Concatenations

    self.agents.group = np.concatenate((self.agents.group, group), axis=0)
    self.agents.vmin = np.concatenate((self.agents.vmin, vmin), axis=0)
    self.agents.vmax = np.concatenate((self.agents.vmax, vmax), axis=0)
    self.agents.rmax = np.concatenate((self.agents.rmax, rmax), axis=0)
    self.agents.vnoise = np.concatenate((self.agents.vnoise, vnoise), axis=0)
    self.agents.dv_scale = np.concatenate((self.agents.dv_scale, dv_scale), axis=0)
    if self.geom.dimension>1:
      self.agents.da_scale = np.concatenate((self.agents.da_scale, da_scale), axis=0)
      self.agents.anoise = np.concatenate((self.agents.anoise, anoise), axis=0)
    if self.geom.dimension>2:
      self.agents.db_scale = np.concatenate((self.agents.db_scale, db_scale), axis=0)
      self.agents.dc_scale = np.concatenate((self.agents.dc_scale, dc_scale), axis=0)
      self.agents.bnoise = np.concatenate((self.agents.bnoise, bnoise), axis=0)
      self.agents.cnoise = np.concatenate((self.agents.cnoise, cnoise), axis=0)

    # --- Groups specifications --------------------------------------------

    self.groups.inputs.append(kwargs['inputs'] if 'inputs' in kwargs else [])
    self.groups.outputs.append(kwargs['outputs'] if 'outputs' in kwargs else [])

  def add_field(self, field, **kwargs):
    '''
    Add a field
    '''

    return self.fields.add(field, **kwargs)

  # ------------------------------------------------------------------------
  #   Setups
  # ------------------------------------------------------------------------

  def setup_animation(self, style='dark', animationClass=None, **kwargs):
    '''
    Define animation
    '''

    self.window = Window('MIDAS', style=style)
    if animationClass is None:
      self.animation = MIDAS.engine.animation.Animation(self, **kwargs)
    else:
      self.animation = animationClass(self, **kwargs)

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

  def set_coefficients(self, i, C):
    '''
    Set the weights
    '''
    self.inputs[i].coefficients = Coefficients(self, i, C)

  def define_parameters(self):
    '''
    Define parameter arrays
    '''

    # --- Geometry ---------------------------------------------------------

    self.param_geometry = self.geom.get_param()

    # --- Agent parameters -------------------------------------------------  

    self.param_agents = self.agents.get_param()

    # --- Input parameters -------------------------------------------------

    l_pparam = []
    field_offset = 0
    for I in self.inputs:

      l_pparam.append(I.get_param(self.geom.dimension, self.groups.N, field_offset))

      # Update field offset
      if I.perception==Perception.FIELD:
        field_offset += self.fields.field[I.field].nSa

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

    # --- Custom parameters ------------------------------------------------

    self.param_custom = np.array([self.custom_param[x] for x in sorted(self.custom_param)])

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
    self.cuda.custom_param = cuda.to_device(self.param_custom.astype(np.float32))
    self.cuda.properties = cuda.to_device(np.zeros((self.agents.N, self.n_CUDA_properties), dtype=np.float32))

    if self.fields is None:
      self.cuda.input_fields = cuda.to_device([np.float32(0)])
    
    # Double buffers
    self.cuda.p0 = cuda.to_device(self.agents.pos.astype(np.float32))
    self.cuda.v0 = cuda.to_device(self.agents.vel.astype(np.float32))
    self.cuda.p1 = cuda.device_array((self.agents.N, self.geom.dimension), np.float32)
    self.cuda.v1 = cuda.device_array((self.agents.N, self.geom.dimension), np.float32)

    # --- Main loop --------------------------------------------------------

    if self.animation is None:

      from alive_progress import alive_it

      ''' It is important that steps start at 1, step=0 being the initial state '''
      bar = alive_it(range(self.steps))
      bar.title = self.verbose.get_caller(1)
      for step in bar:
        if step: self.step(step)

      self.end()

    else:

      # Default information
      if self.information is None:
        self.information = InformationBase(self)

      # Use the animation clock
      self.animation.initialize()
      # self.animation.item[0].colors=('yellow', None)
      self.window.show()

  def step(self, i):

    # Update field inputs
    if self.fields is not None and self.fields.N:      

      # Perceive field
      self.cuda.input_fields = cuda.to_device(np.array(self.fields.perception()).astype(np.float32))

      # Update field
      self.fields.update()

    else:
      self.cuda.input_fields = cuda.to_device(np.empty(0, dtype=np.float32))
    
    # Double-buffer computation trick
    if i % 2:
      
      self.cuda.step[self.cuda.gridDim, self.cuda.blockDim](self.cuda.geometry,
        self.cuda.agents, self.cuda.perceptions, self.cuda.actions, self.cuda.groups,
        self.cuda.custom_param, self.cuda.input_fields, self.cuda.properties,
        self.cuda.p0, self.cuda.v0, self.cuda.p1, self.cuda.v1,
        self.cuda.rng)
      
      # Get back position and velocities
      cuda.synchronize()      
      self.agents.pos = self.cuda.p1.copy_to_host()
      self.agents.vel = self.cuda.v1.copy_to_host()

    else:

      self.cuda.step[self.cuda.gridDim, self.cuda.blockDim](self.cuda.geometry,
        self.cuda.agents, self.cuda.perceptions, self.cuda.actions, self.cuda.groups,
        self.cuda.custom_param, self.cuda.input_fields, self.cuda.properties,
        self.cuda.p1, self.cuda.v1, self.cuda.p0, self.cuda.v0,
        self.cuda.rng)
      
      # Get back position and velocities
      cuda.synchronize()
      self.agents.pos = self.cuda.p0.copy_to_host()
      self.agents.vel = self.cuda.v0.copy_to_host()
    
    if self.n_CUDA_properties:
      self.properties = self.cuda.properties.copy_to_host()

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
    nstep = (self.window.step if self.animation is not None else self.steps)+1
    self.verbose(f'End of simulation @ {nstep:d} steps ({time.time()-self.tref:.2f} s)')
    self.verbose.line()

    # End storage
    if self.storage is not None:
      self.storage.db_conn.commit()

    # End display
    if self.animation is not None:
      self.animation.is_running = False
      self.animation.window.close()

############################################################################
# ######################################################################## #
# #                                                                      # #
# #                                                                      # #
# #                               CUDA                                   # #
# #                                                                      # #
# #                                                                      # #
# ######################################################################## #
############################################################################

'''
Vocabulary:
  - A grid is a polar grid, defined for each perception function
  - A grid is composed of zones
  - A set of inputs in a griven grid is called a 'perception'

=== PARAMETERS =============================================================

On the cuda side the general model is decomposed in different parameter sets:

      [Geometry parameters]   (1 row)
geometry
  ├── dimension         (1)   dimension
  ├── arena             (1)   arena type
  ├── X-size            (1)   arena size in the 1st dimension
  ├── Y-size    [dim>1] (1)   arena size in the 2nd dimension
  ├── Z-size    [dim>2] (1)   arena size in the 3rd dimension
  ├── X-period          (1)   arena periodicity in the 1st dimension
  ├── Y-period  [dim>1] (1)   arena periodicity in the 2nd dimension
  └── Z-period  [dim>2] (1)   arena periodicity in the 3rd dimension

          [Agent parameters]  (N rows)
agents
  ├── group             (1)   group index
  ├── vlim              (2)   speed limits (vmin, vmax)
  ├── rmax              (1)   maximal distance for visibility
  ├── dv_scale          (1)   speed modulation scale
  ├── da_scale  [dim>1] (1)   ┐ 
  ├── db_scale  [dim>2] (1)   │ reorientation scale
  ├── dc_scale  [dim>2] (1)   ┘
  ├── vnoise            (1)   speed noise
  ├── anoise    [dim>1] (1)   ┐
  ├── bnoise    [dim>2] (1)   │ reorientation noises
  └── cnoise    [dim>2] (1)   ┘

           [Input parameters] (nP rows)
perceptions
  ├── ptype           (1)     perception type
  ├── ntype           (1)     normalization type
  ├── foffset         (1)     0 or field offset [if ptype==Perception.FIELD]
  ├── nR              (1)     ┐ number of radial region, nR = len(rZ) + 1
  ├── rmax            (1)     │
  ├── nSa     [dim>1] (1)     │ Grid definition
  ├── nSb     [dim>2] (1)     │
  ├── rZ              (nR-1)  ┘
  ├── nW              (1)     number of weights
  └── weights         (var)   ┐
      ├── w0          (1)     │ as many as weights
      └── ...                 ┘

          [Output parameters] (nP rows)
actions
  ├── otype           (1)     output type
  └── ftype           (1)     activation type

          [Group parameters]  (nG rows)
groups
  ├── atype           (1)     type of the agents in the group
  ├── nP              (1)     number of perceptions
  ├── nO              (1)     number of outputs
  ├── perceptions     (var)   ┐
  │   ├── p0          (1)     │ as many as perceptions
  │   └── ...                 ┘
  └── outputs         (var)   ┐
      ├── o0          (1)     │ as many as outputs
      └── ...                 ┘

      [Custom parameters]  (1 row)
From parameters in the engine.custom dict, with keys ordered alphabetically.

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

=== FIELD INPUTs ===========================================================

Fields are managed on the host (CPU) side. The field-related input are computed 
on the host and sent directly to the device in the 'input_fields' array. The number
of columns per field is defined by the corresponding perception nSa.

      [Field inputs]    (N row)
input_fields
  ├── field 0         (var)   ┐
  │   ├── in_0        (1)     │ Inputs of field 0
  │   └── ...                 ┘
  ...                           ...
  └── field Nf-1      (var)   ┐
      ├── in_0        (1)     │ Inputs of field Nf-1
      └── ...                 ┘
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
    self.input_fields = []
    self.custom_param = None
    self.properties = None
    
    # Random number generator
    self.rng = create_xoroshiro128p_states(self.blockDim*self.gridDim, seed=0)

    # --------------------------------------------------------------------------
    #   CUDA kernel variables
    # --------------------------------------------------------------------------

    # Number of agents
    N = self.engine.agents.N

    # Agent-driven perception boolean
    agent_drivenity = self.engine.agent_drivenity

    # CUDA local array dimensions
    dim  = self.engine.geom.dimension
    nI = int(np.sum([x.weights.size for x in self.engine.inputs]))
    m_nIpp = max([x.weights.size for x in self.engine.inputs])
    m_nO = max([len(x) for x in self.engine.groups.outputs])
   
    # -- Customizable CUDA functions

    # Perception
    if self.engine.CUDA_perception is None:
      import MIDAS.CUDA.perception as perception
    else:
      perception = importlib.import_module(self.engine.CUDA_perception)

    # Action
    if self.engine.CUDA_action is None:
      import MIDAS.CUDA.action as action
    else:
      action = importlib.import_module(self.engine.CUDA_action)
            
    # --------------------------------------------------------------------------
    #   The CUDA kernel
    # --------------------------------------------------------------------------
    
    @cuda.jit(cache=False)
    def CUDA_step(geometry, agents, perceptions, actions, groups, custom_param, input_fields, properties, p0, v0, p1, v1, rng):
      '''
      The CUDA kernel
      '''

      i = cuda.grid(1)

      if i<p0.shape[0]:
                
        # Group id
        gid = int(agents[i, 0])

        # Agent type
        atype = int(groups[gid, 0])
        
        # === Fixed points =================================================

        if atype==Agent.FIXED.value:
          for j in range(dim):
            p1[i,j] = p0[i,j]
            v1[i,j] = v0[i,j]
          return
        
        # === Mobile points ================================================
        
        # --- Definitions --------------------------------------------------

        # --- Agent properties

        agent = cuda.local.array(6, nb.float32)

        # Agent
        agent[0] = i
        agent[1] = gid
        agent[2] = p0[i,0]
        agent[3] = p0[i,1]
        agent[4] = v0[i,0]
        agent[5] = v0[i,1]

        z0 = complex(p0[i,0], p0[i,1])

        # --- Other properties

        nG = groups.shape[0]
        nP = groups[gid,1]
        nO = groups[gid,2]

        # --- Other agents relative coordinates

        if agent_drivenity:
          
          # Visibility limit
          rmax = agents[i,3]

          z = cuda.local.array(N, nb.complex64)
          alpha = cuda.local.array(N, nb.float32)
          visible = cuda.local.array(N, nb.boolean)

          for j in range(N):

            # Skip self-perception
            if j==i:
              visible[j] = False
              continue

            # Distance and relative orientation
            match dim:
              case 1: pass
              case 2:
                z[j], alpha[j], visible[j] = relative_2d(p0[i,0], p0[i,1], v0[i,1], p0[j,0], p0[j,1], v0[j,1], rmax, geometry)
              case 3: pass

        else:
          z = cuda.local.array(1, nb.complex64)
          alpha = cuda.local.array(1, nb.float32)
          visible = cuda.local.array(1, nb.boolean)

        if m_nO>0:

          '''
          Parameters are fixed, they cannot be altered in the perception function
          '''
          param = (geometry, groups, agents, perceptions, actions,
                  agent, z, alpha, visible, 
                  input_fields, custom_param)

          # Output vector
          vOut = cuda.local.array(m_nO, nb.float32)
          
          # === INPUTS

          # Inputs local array
          vIn = cuda.local.array(nI, nb.float32)
          pIn = cuda.local.array(m_nIpp, nb.float32)
          vi = 0
          
          for pi in range(nP):

            # Perception index
            p = int(groups[gid, pi+3])

            # Field offset
            field_offset = int(perceptions[p,2])

            # Grid parameters
            nR = perceptions[p,3]
            nSa = perceptions[p,5] if dim>1 else 1
            nSb = perceptions[p,6] if dim>2 else 1

            # Number of inputs
            nIpp = int(nG*nR*nSb*nSa)

            # --- Define parameters
            '''
            Parameters are fixed, they cannot be altered in the perception function)
            '''
            
            pparam = (m_nIpp, nO, nG, nR, nSa, nSb, field_offset)

            # === Inputs

            # Reset input buffer
            if pi>0: 
              for k in range(nIpp): pIn[k] = 0
            
            # Perception function
            pIn, properties, rng = perception.perceive(pIn, properties, rng, p, param, pparam)

            # === Normalization

            pIn = normalize(pIn, perceptions[p,1], pparam)

            # === Storage

            for k in range(nIpp):
              vIn[vi+k] = pIn[k]
            vi += nIpp

          # === OUTPUTS

          match atype:
            
            case Agent.RIPO.value:
              vOut, properties = MIDAS.network.run(vOut, properties, vIn, param)

          # --- Activations

          for oid in range(nO):

            aid = int(groups[gid, int(nP + 3 + oid)])
            ftype = actions[aid,1]

            # --- Activation

            match ftype:

              case Activation.IDENTITY.value:
                pass

              case Activation.HSM_POSITIVE.value:
                vOut[oid] = 2/math.pi*math.atan(math.exp(vOut[oid]/2))

              case Activation.HSM_CENTERED.value:
                # output = 4/math.pi*math.atan(math.exp((vOut[oid])/2))-1
                vOut[oid] = 4/math.pi*math.atan(math.exp(vOut[oid]/2))-1
                  
          # === Actions (velocity updates)

          V = cuda.local.array(dim, nb.float32)
          for d in range(dim): V[d] = v0[i,d]

          V = action.update_velocities(V, vOut, param, properties, rng)

          # === Update positions

          # Boundary conditions
          p1[i,0], p1[i,1], v1[i,0], v1[i,1] = assign_2d(z0, V, geometry)

    # Store CUDA kernel
    self.step = CUDA_step

# --------------------------------------------------------------------------
#   Boundary conditions
# --------------------------------------------------------------------------

@cuda.jit(device=True, cache=True)
def relative_2d(x0, y0, a0, x1, y1, a1, rmax, geometry):
  '''
  Relative position and orientation between two agents
  The output is tuple (z, alpha, status) containing the relative complex polar
  coordinates z, the relative orientation alpha and the visibility status.
  NB: in case the distance is above rmax, (0,0,False) is returned
  '''

  # --- Definitions
  
  dim = geometry[0]
  arena = geometry[1]

  match dim:

    case 1:

      # Arena shape
      arena_X = geometry[2]

      # Arena periodicity
      periodic_X = geometry[3]

    case 2:

      # Arena shape
      arena_X = geometry[2]
      arena_Y = geometry[3]

      # Arena periodicity
      periodic_X = geometry[4]
      periodic_Y = geometry[5]

    case 3:

      # Arena shape
      arena_X = geometry[2]
      arena_Y = geometry[3]
      arena_Z = geometry[4]

      # Arena periodicity
      periodic_X = geometry[5]
      periodic_Y = geometry[6]
      periodic_Z = geometry[7]

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
  if rmax>0 and abs(z)>rmax: return (0, 0, False)

  # Orientation
  z *= cmath.rect(1., -a0)

  return (z, a1-a0, True)

@cuda.jit(device=True, cache=True)
def assign_2d(z0, V, geometry):
  
  # --- Definitions
  
  dim = geometry[0]
  arena = geometry[1]

  match dim:

    case 1:

      # Arena shape
      arena_X = geometry[2]

      # Arena periodicity
      periodic_X = geometry[3]

    case 2:

      # Arena shape
      arena_X = geometry[2]
      arena_Y = geometry[3]

      # Arena periodicity
      periodic_X = geometry[4]
      periodic_Y = geometry[5]

    case 3:

      # Arena shape
      arena_X = geometry[2]
      arena_Y = geometry[3]
      arena_Z = geometry[4]

      # Arena periodicity
      periodic_X = geometry[5]
      periodic_Y = geometry[6]
      periodic_Z = geometry[7]

  v = V[0]
  a = V[1]

  # --- Computations

  # Candidate position and velocity
  z1 = z0 + cmath.rect(v, a)

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

    # Sanity check
    if abs(z1) > arena_X: z1 = z1/abs(z1)*(arena_X-1e-6)

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

@cuda.jit(device=True, cache=True)
def normalize(vIn, ntype, pparam):

  nG = pparam[ip_NG]
  nR = pparam[ip_NR]
  nSa = pparam[ip_NSA]
  nSb = pparam[ip_NSB]

  match ntype:

    case Normalization.SAME_RADIUS.value:
      '''
      Normalization over the same radius
      '''

      for ig in range(nG):
        for ir in range(nR):

          # Get sum
          S = 0
          for k in range(nSb*nSa):
            S += vIn[int(ig*nR*nSb*nSa + ir*nSb*nSa + k)]

          # Normalization
          if S!=0:
            for k in range(nSb*nSa):
              vIn[int(ig*nR*nSb*nSa + ir*nSb*nSa + k)] /= S

    case Normalization.SAME_SLICE.value:
      '''
      Normalization over the same angular slice
      '''
      
      for ig in range(nG):
        for ia in range(nSb*nSa):

          # Get sum
          S = 0
          for k in range(nR):
            S += vIn[int(ig*nR*nSb*nSa + k*nSb*nSa + ia)]

          # Normalization
          if S!=0:
            for k in range(nR):
              vIn[int(ig*nR*nSb*nSa + k*nSb*nSa + ia)] /= S

    case Normalization.SAME_GROUP.value:
      '''
      Normalization over all zones of the same group
      '''

      for ig in range(nG):

        # Get sum
        S = 0
        for k in range(nR*nSb*nSa):
          S += vIn[int(ig*nR*nSb*nSa + k)]

        # Normalization
        if S!=0:
          for k in range(nR*nSb*nSa):
            vIn[int(ig*nR*nSb*nSa + k)] /= S
      
    case Normalization.ALL.value:
      '''
      Normalization over all zones of all groups
      '''

      # Get sum
      S = 0
      for k in range(nG*nR*nSb*nSa):
        S += vIn[k]

      # Normalization
      if S!=0:
        for k in range(nG*nR*nSb*nSa):
          vIn[k] /= S

  return vIn