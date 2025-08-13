'''
MIDAS Engine
'''

import numpy as np
from rich import print
from rich.panel import Panel
from rich.table import Table

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

  # ────────────────────────────────────────────────────────────────────────
  def rich(self):
    
    # ─── Groups

    groups = '[b i #abcdef]groups[/]\n'
    if not len(self.group):
      groups += '─── [i]no group[/]'

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

    # Initialize animation
    self._animation.initialize()

# W = anim.window('Line animation')

# # Add animation
# W.add(Canva)

# # Allow backward animation
# W.allow_backward = True
# W.allow_negative_time = True

# W.show()