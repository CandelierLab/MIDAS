'''
ANN input
'''

import numpy as np
from rich import print
from rich.panel import Panel

import MIDAS

class input:

  # ────────────────────────────────────────────────────────────────────────
  def __init__(self, group, itype, canva,
               perceived = MIDAS.GROUP.SELF,
               weights = None,
               coefficients = None,
               normalization = MIDAS.NORMALIZATION.NONE):

    # Group
    self.group = group

    # Canva
    self.canva = canva

    # Coefficients
    if weights is not None:
      self.weights = weights
    else:

      match coefficients.ndim:
        case 1:
          nc = int(coefficients.size/2)
          coefficients[nc:] *= -1

        case 2:
          nc = int(coefficients.shape[1]/2)
          coefficients[:,nc:] *= -1

      self.weights = coefficients
      
  # ────────────────────────────────────────────────────────────────────────
  def rich(self, title=True):
    
    s = ''

    if title:
      s += f'[b i #abcdef]input[/]\n'

    return s

  # ────────────────────────────────────────────────────────────────────────
  def display(self): 

    print(Panel(self.rich(title=False), title=f'input'))
