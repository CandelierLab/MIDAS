'''
MIDAS Engine
'''

# import importlib
# import warnings
# import math, cmath
# import time
import numpy as np

import MIDAS

class engine:
  '''
  Engine
  '''

  # ════════════════════════════════════════════════════════════════════════
  #                               CONSTRUCTOR
  # ════════════════════════════════════════════════════════════════════════
    
  def __init__(self, dimension=2, **kwargs):
    '''
    Constructor

    Initializes the geometry and the agents
    '''

    # --- Initialization

    # self.geom = Geometry(dimension, **kwargs)
    # self.agents = Agents(dimension)
    # self.groups = Groups(dimension)
    # self.fields = Fields(self)
    # self.inputs = []
    # self.outputs = []
    
    # # Storage
    # self.storage = None

    # # Animation
    # self.window = None
    # self.animation = None
    # self.information = None

    # # --- GPU

    # self.cuda = None        

    # # Parameters for the kernel
    # self.param_geometry = None
    # self.param_agents = None
    # self.param_perceptions = None
    # self.param_outputs = None
    # self.param_groups = None
    # self.param_custom = None

    # # CUDA variables
    # self.agent_drivenity = False

    # # --- Customizable CUDA packages

    # self.CUDA_perception = None
    # self.CUDA_action = None

    # self.n_CUDA_properties = 0
    # self.properties = None

    # # --- Time

    # # Total number of steps
    # self.steps = None

    # # Computation time reference
    # self.tref = None

    # # --- Misc attributes
    
    # self.custom_param = {}
    # self.verbose = MIDAS.verbose.cli_Reporter()
    # self.verbose.level = Verbose.NORMAL


