'''
MIDAS core - Agents
'''

import numpy as np
from rich import print
from rich.panel import Panel

import MIDAS

# ══════════════════════════════════════════════════════════════════════════
class agents:

  # ────────────────────────────────────────────────────────────────────────
  def __init__(self):
    '''
    The collection of agents is initially empty.
    Agents have to be added via the MIDAS.engine.add_group method.
    '''

    # Number of agents
    self.N = 0

    # Groups
    self.group = np.array([], dtype=int)

    # Positions and velocities
    '''
    Position are expressed in cartesian coordinates (x,y,z)
    Velocities are expressed in polar coordinates (v,alpha,beta)
    '''
    self.x = np.array([])
    self.y = np.array([])
    self.v = np.array([])
    self.a = np.array([])

  # ────────────────────────────────────────────────────────────────────────
  def rich(self, title=True):

    s = ''

    if title:
      s += '[b i #abcdef]agents[/]\n'

    # Empty set
    if not self.N:
      s += '─── [i]no agent[/]'
      return s
    
    s += f'{self.N} agents'

    return s

  # ────────────────────────────────────────────────────────────────────────
  def display(self): 
    
    print(Panel(self.rich(title=False), title='agents'))

  # ────────────────────────────────────────────────────────────────────────
  def pos(self, i): return [self.x[i], self.y[i]]

  # ────────────────────────────────────────────────────────────────────────
  def vel(self, i): return [self.v[i], self.a[i]]