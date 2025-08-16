'''
ANN input
'''

import numpy as np
from rich import print
from rich.panel import Panel

import MIDAS

class input:

  # ────────────────────────────────────────────────────────────────────────
  def __init__(self, group, itype, 
               perceived = MIDAS.GROUP.SELF, 
               canva = None,
               coefficients = None,
               normalization = MIDAS.NORMALIZATION.NONE):

    # Group
    self.group = group

    # Canva
    self.canva = canva
    if canva is None:
      self.canva = MIDAS.ANN.spatial(self.group, [], 1, 1) if self.group.canva is None else self.group.canva
      
  # ────────────────────────────────────────────────────────────────────────
  def rich(self, title=True):
    
    s = ''

    if title:
      s += f'[b i #abcdef]input[/]\n'

    return s

  # ────────────────────────────────────────────────────────────────────────
  def display(self): 

    print(Panel(self.rich(title=False), title=f'input'))
