'''
MIDAS agents driven by perceptrons
'''

import numpy as np
from rich import print
from rich.panel import Panel
from rich.columns import Columns

import MIDAS

# ══════════════════════════════════════════════════════════════════════════
class perceptron:

  # ════════════════════════════════════════════════════════════════════════
  #                             Initialization
  # ════════════════════════════════════════════════════════════════════════

  # ────────────────────────────────────────────────────────────────────────
  def __init__(self, engine, number, name='perceptron'):
    '''
    The collection of agents is initially empty.
    Agents have to be added via the engine.add_group method.
    '''

    # Engine
    self.engine = engine

    # Number of agents
    self.N = number

    # Group name
    self.name = name

    # ─── Declare group

    # Group id
    self.id = None

    # Locations in agents.pos and agents.vel
    self.Id = None

    # Add the group
    self.engine.add_group(self)

    # Initial conditions
    self.initial = MIDAS.core.initial_conditions(self)

    # ─── Inputs

    # Input list
    self.l_input = []

    # ─── Dynamics

    self.vmin = 0
    self.vmax = np.inf
    
  # ════════════════════════════════════════════════════════════════════════
  #                                Display
  # ════════════════════════════════════════════════════════════════════════

  # ────────────────────────────────────────────────────────────────────────
  def rich(self, title=True, oneline=False):

    # ─── One-liner

    if oneline:
      return f'[i #555555][{self.id}][/] {self.name} ([cyan]{self.N}[/])'
    
    # ─── Expanded display

    s = ''

    if title:
      s += f'[b i #abcdef]{self.name}[/]\n'
    
    s += f'group id: {self.id}\n'
    s += f'{self.N} agents'

    return s

  # ────────────────────────────────────────────────────────────────────────
  def display(self):
    
    print(Panel(Columns([self.rich(), self.initial.rich()],
                        equal=True, expand=True),
                title=self.name))
    
  # ════════════════════════════════════════════════════════════════════════
  #                                  I/O
  # ════════════════════════════════════════════════════════════════════════

  # ────────────────────────────────────────────────────────────────────────
  def input(self, itype, canva,
            perceived = MIDAS.GROUP.SELF,
            coefficients = None,
            normalization = MIDAS.NORMALIZATION.NONE):
    
    # New input
    input = MIDAS.ANN.input(self, itype, canva,
                            perceived = perceived,
                            coefficients = coefficients,
                            normalization = normalization)

    # Append input
    self.l_input.append(input)

    return input

    # # Field grids
    # match perception:
    #   case Perception.FIELD:
    #     nSa = kwargs['nSa'] if 'nSa' in kwargs else 4
    #     kwargs['grid'] = PolarGrid(nSa=nSa)

    # # Append input
    # self.inputs.append(Input(perception, **kwargs))

    # # Check agent-drivenity
    # if ('agent_drivenity' in kwargs and kwargs['agent_drivenity']) \
    #   or perception in [Perception.PRESENCE, Perception.ORIENTATION]:

    #   self.agent_drivenity = True

    # return len(self.inputs)-1
