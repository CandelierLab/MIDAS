'''
MIDAS Engine
'''

import os
import warnings
import numpy as np
from rich import print
from rich.panel import Panel
from rich.table import Table
import pyopencl as cl
from pyopencl.clrandom import PhiloxGenerator
import pyopencl.array as cl_array

os.environ['PYOPENCL_CTX'] = '0'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

import MIDAS

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
# █░░░░░░░░░░░░░░░░░░░░░░░░░░ ENGINE FRONTEND ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
# █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

class engine:
  '''
  Engine
  '''

  # ════════════════════════════════════════════════════════════════════════
  #                               CONSTRUCTOR
  # ════════════════════════════════════════════════════════════════════════
    
  # ────────────────────────────────────────────────────────────────────────
  def __init__(self, dimension=2, platform='GPU', multi=1, **kwargs):
    '''
    Constructor

    Initializes the geometry and the agents
    '''

    # ─── Initialization

    self.geometry = MIDAS.core.geometry(dimension, **kwargs)
    
    # ─── Time

    # Total number of steps
    self.steps = None

    # Computation time reference
    self.tref = None

    # ─── Agents

    self.group = []
    self.agents = MIDAS.core.agents()
    
    # ─── Fields

    self.fields = MIDAS.core.fields()

    # ─── Animation

    # self.window = None
    self._animation = None
    # self.information = None

    # ─── Storage

    self.storage = None

    # ─── Platform

    self.platform = platform
    self.multi = multi
    
  # ════════════════════════════════════════════════════════════════════════
  #                                PROPERTIES
  # ════════════════════════════════════════════════════════════════════════
   
  # ─── animation ──────────────────────────────────────────────────────────
  
  @property
  def animation(self): return self._animation

  @animation.setter
  def animation(self, a):

    # Define animation
    self._animation = a

    # Define engine
    self._animation.engine = self

    # Aniation initialization
    self._animation.initialize()

  # ════════════════════════════════════════════════════════════════════════
  #                                 DISPLAY
  # ════════════════════════════════════════════════════════════════════════
  
  # ────────────────────────────────────────────────────────────────────────
  def rich(self):
    
    # ─── Groups

    groups = '[b i #abcdef]groups[/]\n'
    if not len(self.group):
      groups += '─── [i]no group[/]'
    else:
      for group in self.group:
        groups += '• ' + group.rich(oneline=True)

    # ─── Simulation

    simulation = '[b i #abcdef]simulation[/]\n'
    simulation += '─── [i]no step limit[/]' if self.steps is None else f'{self.steps} steps'

    grid = Table.grid(expand=True, padding=1)
    grid.add_column()
    grid.add_column(justify="left")
    grid.add_row(self.geometry.rich(), self.fields.rich())
    grid.add_row(groups, self.agents.rich())
    grid.add_row(simulation, '')

    return grid

  # ────────────────────────────────────────────────────────────────────────
  def display(self): 

    print(Panel(self.rich(), title='engine'))

  # ════════════════════════════════════════════════════════════════════════
  #                                 Agents
  # ════════════════════════════════════════════════════════════════════════
  
  # ────────────────────────────────────────────────────────────────────────
  def add_group(self, group):
    '''
    Add a group of agents

    Args:
        group (_type_): _description_

    Returns:
        _type_: _description_
    '''

    # Append group
    self.group.append(group)

    # Set group id
    group.id = len(self.group)-1

    # ─── Update agents list

    I0 = self.agents.N

    # Update number of agents
    self.agents.N += group.N

    # Update groups
    self.agents.group = np.concatenate((self.agents.group,
                                        np.full(group.N, fill_value=group.id)))
    
    # Set identifiers
    group.Id = np.array(range(I0, self.agents.N), dtype=int)

  # ════════════════════════════════════════════════════════════════════════
  #                               Simulation
  # ════════════════════════════════════════════════════════════════════════
  
  # ────────────────────────────────────────────────────────────────────────
  def run(self):

    # ─── Checks ────────────────────────────────

    # No animation
    if self.animation is None:
    
      # Number of steps
      if self.steps is None:
        warnings.warn('The number of steps must be defined when there is no animation.')
        return
      
      # # Storage
      # if self.storage is None:
      #   warnings.warn('A storage location must be defined when there is no animation.')
      #   return

    # ─── Initial conditions ────────────────────

    # Initialize arrays
    self.agents.x = np.empty(self.agents.N)
    self.agents.y = np.empty(self.agents.N)
    self.agents.v = np.empty(self.agents.N)
    self.agents.a = np.empty(self.agents.N)    

    for group in self.group:

      # Get initial positions
      self.agents.x[group.Id], self.agents.y[group.Id] = group.initial.get_positions()

      # Get initial velocities
      self.agents.v[group.Id], self.agents.a[group.Id] = group.initial.get_velocities()

    # ─── Platform engine ───────────────────────

    match self.platform:
      case 'CPU': self.cpu = CPU(self)
      case 'GPU': self.gpu = GPU(self)

    # ─── Storage  ──────────────────────────────

    # ─── Timing  ───────────────────────────────

    # ═══ Main loop  ════════════════════════════

    if self.animation is None:

      from alive_progress import alive_it

      ''' It is important that steps start at 1, step=0 being the initial state '''
      bar = alive_it(range(self.steps))
      bar.title = self.verbose.get_caller(1)
      for step in bar:
        if step: self.step(step)

      self.end()

    else:

      # GPU imports
      if self.platform=='GPU':
        self.gpu.import_position = True
        self.gpu.import_velocity = True

      # Use the animation clock
      self.animation.initial_setup()
      self.animation.window.show()

  # ────────────────────────────────────────────────────────────────────────
  def step(self, i):
    '''
    Operations to do at evey step
    '''

    # ─── Update velocities ─────────────────────

    match self.platform:

      case 'CPU': 
        # TODO
        pass

      case 'GPU': 
        
        # Build inputs
        self.gpu.perception()

        # Compute outputs
        self.gpu.computation()

        # Update positions
        self.gpu.motion()
    
  # ────────────────────────────────────────────────────────────────────────
  def end(self):
    '''
    Operations to do when the simalutation is over
    '''

    pass

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
# █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ CPU ENGINE ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
# █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

class CPU:
  
  # ════════════════════════════════════════════════════════════════════════
  #                               CONSTRUCTOR
  # ════════════════════════════════════════════════════════════════════════
    
  # ────────────────────────────────────────────────────────────────────────
  def __init__(self, engine):
    '''
    Constructor
    '''

    # Engine
    self.engine = engine

    # Multiverses
    self.multi = self.engine.multi

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
# █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ GPU ENGINE ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
# █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

class GPU:
  
  # ════════════════════════════════════════════════════════════════════════
  #                               CONSTRUCTOR
  # ════════════════════════════════════════════════════════════════════════
    
  # ────────────────────────────────────────────────────────────────────────
  def __init__(self, engine):
    '''
    Constructor
    '''

    # Engine
    self.engine = engine

    # Multiverses
    self.multi = self.engine.multi
    ''' NB: Multiverses are not implemented yet '''

    # Kernels
    self.kernel = type('obj', (object,), {'perception': None,
                                          'computation': None,
                                          'motion': None})

    # ─── Import options ────────────────────────

    self.import_position = False
    self.import_velocity = False

    # ─── Types ─────────────────────────────────

    self.type_idx = np.uint32       # Indices
    self.type_pos = np.float32      # Positions
    self.type_vel = np.float32      # Velocities
    self.type_cnv = np.float32      # Canva
    self.type_cff = np.float32      # Coefficients

    # ─── Constants ─────────────────────────────

    # Number of agents
    # # # self.n_agents = self.n_agents_format(self.engine.agents.N)

    # ─── OpenCL machinery ──────────────────────

    # ─── Context

    platform = cl.get_platforms()
    my_gpu_devices = [platform[0].get_devices(device_type=cl.device_type.GPU)[0]]
    ctx = cl.Context(devices=my_gpu_devices)

    # Queue
    self.queue = cl.CommandQueue(ctx)

    # Random number generator
    self.rng =  PhiloxGenerator(context=ctx)

    # ─── Kernels ───────────────────────────────

    # ─── Perception

    '''
    This kernel builds the inputs for each agent
    It handles the boundary conditions (bouncing and periodic)
    '''

    prg = cl.Program(ctx, open(MIDAS.path + 'GPU' + os.path.sep + 'perception_2d.cl').read()).build()
    self.kernel.perception = prg.perception

    # ─── Computation

    '''
    This kernel computes the outputs for each agent
    '''

    prg = cl.Program(ctx, open(MIDAS.path + 'GPU' + os.path.sep + 'computation.cl').read()).build()
    self.kernel.computation = prg.computation

    # ─── Motion

    '''
    This kernel sets the new positions and orientation based on the velocities.
    It handles the boundary conditions (bouncing and periodic)
    '''

    prg = cl.Program(ctx, open(MIDAS.path + 'GPU' + os.path.sep + 'motion_2d.cl').read()).build()
    self.kernel.motion = prg.motion

    # ─── Data ──────────────────────────────────
    
    # ─── Host arrays

    self.engine.agents.x = self.engine.agents.x.astype(self.type_pos)
    self.engine.agents.y = self.engine.agents.y.astype(self.type_pos)
    self.engine.agents.v = self.engine.agents.v.astype(self.type_vel)
    self.engine.agents.a = self.engine.agents.a.astype(self.type_vel)

    # ─── Device arrays

    mf = cl.mem_flags

    # Arena
    match self.engine.geometry.arena.type:

      case MIDAS.ARENA.CIRCULAR:
        arena = np.array([self.engine.geometry.arena.type,
                          self.engine.geometry.arena.radius], dtype = np.float32)
        
      case MIDAS.ARENA.RECTANGULAR:
        arena = np.array([self.engine.geometry.arena.type,
                          self.engine.geometry.arena.shape[0]/2,
                          self.engine.geometry.arena.shape[1]/2], dtype = np.float32)
        
    self.d_arena = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = arena)

    # Boundary conditions
    bcond = np.array(self.engine.geometry.arena.periodic, dtype = bool)
    self.d_bcond = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = bcond)

    # Position and velocity
    self.d_x = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.engine.agents.x)
    self.d_y = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.engine.agents.y)
    self.d_v = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.engine.agents.v)
    self.d_a = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.engine.agents.a)
    
    # Interactions
    pairs = np.stack(np.triu_indices(self.engine.agents.N,1)).astype(self.type_idx).T.copy()
    self.d_pairs = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = pairs)
    self.n_pairs = pairs.shape[0]

    # Canva
    canva = self.engine.group[0].l_input[0].canva
    cnv_r, cnv_theta = canva.prepare(self.type_cnv)
    self.d_cnv_r = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = cnv_r)
    self.d_cnv_theta = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = cnv_theta)

    # Input
    self.d_input = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                             hostbuf = np.zeros((self.engine.agents.N, canva.nR, canva.nAz), dtype=np.uint32))
    
    # DEBUG /!\ used in self.perception
    self.h_input = np.zeros((self.engine.agents.N, canva.nR, canva.nAz), dtype=np.uint32)    

    # Coefficients
    self.d_coeff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                             hostbuf = self.engine.group[0].l_input[0].coefficients.astype(self.type_cff))

    # Random numbers
    # self.d_rnd = cl_array.zeros(self.queue, self.multi*self.n_agents, dtype=np.float32)

  # ────────────────────────────────────────────────────────────────────────
  def perception(self):
    '''
    Build the agents inputs
    '''
    
    # Reset inputs to zeros
    cl.enqueue_fill_buffer(self.queue, self.d_input, np.uint32(0), 0, self.h_input.nbytes)

    self.kernel.perception(self.queue, [self.n_pairs], None,     # Required arguments
                       self.d_arena,
                       self.d_bcond,
                       self.d_pairs,
                       self.d_x,           # ┐
                       self.d_y,           # ┘ position
                       self.d_a,           # - velocity
                       self.d_cnv_r,
                       self.d_cnv_theta,
                       self.d_input,
                       ).wait()
    
    # DEBUG
    # cl.enqueue_copy(self.queue, self.h_input, self.d_input)
    # print(self.h_input)

  # ────────────────────────────────────────────────────────────────────────
  def computation(self):
    '''
    Computation of the agents ouputs
    '''

    self.kernel.computation(self.queue, [self.engine.agents.N], None,     # Required arguments
                            self.d_input,
                            self.d_cnv_r,
                            self.d_cnv_theta,                       
                            self.d_coeff,
                            self.d_a,
                            ).wait()

  # ────────────────────────────────────────────────────────────────────────
  def motion(self):
    '''
    Update the agents position and orientations
    '''

    # # Prepare random array
    # self.rng.fill_uniform(self.d_rnd)

    self.kernel.motion(self.queue, [self.engine.agents.N], None,     # Required arguments
                       self.d_arena,
                       self.d_bcond,
                       self.d_x,    # ┐
                       self.d_y,    # ┘ position
                       self.d_v,    # ┐
                       self.d_a,    # ┘ velocity
                       ).wait()

    # ─── Imports

    if self.import_position:    
      cl.enqueue_copy(self.queue, self.engine.agents.x, self.d_x)
      cl.enqueue_copy(self.queue, self.engine.agents.y, self.d_y)

    if self.import_velocity:
      cl.enqueue_copy(self.queue, self.engine.agents.v, self.d_v)
      cl.enqueue_copy(self.queue, self.engine.agents.a, self.d_a)
