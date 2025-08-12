'''
MIDAS core - Group of agents
'''

import numpy as np
from rich import print
from rich.panel import Panel

import MIDAS

# ══════════════════════════════════════════════════════════════════════════
class group:

  # ────────────────────────────────────────────────────────────────────────
  def __init__(self):
    
    # Groups
    self.name = None

    # Limits
    self.vmin = None
    self.vmax = None
    self.rmax = None

    # Noise
    # self.vnoise = np.empty(0)
    # if self.dimension>1: self.anoise = np.empty(0)
    # if self.dimension>2: 
    #   self.bnoise = np.empty(0)
    #   self.cnoise = np.empty(0)

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

  # ────────────────────────────────────────────────────────────────────────
  def rich(self):
    
    s = f'{self.dimension}d [cyan]'

    match self.type:
      case MIDAS.ARENA.RECTANGULAR: s += 'rectangular'
      case MIDAS.ARENA.CIRCULAR: s += 'circular'

    s += '[/cyan] arena\n'

    match self.type:
      case MIDAS.ARENA.RECTANGULAR: s += f'shape:       {self.shape}\n'
      case MIDAS.ARENA.CIRCULAR: s += f'radius:       {self.radius}\n'
    
    s += f'periodicity: {self.periodic}'
    return s

  # ────────────────────────────────────────────────────────────────────────
  def display(self): print(Panel(self.rich(), title='arena'))