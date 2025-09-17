'''
MIDAS core - Fields
'''

import warnings
import numpy as np
from rich import print
from rich.panel import Panel

import MIDAS

# ══════════════════════════════════════════════════════════════════════════
class fields:

  # ────────────────────────────────────────────────────────────────────────
  def __init__(self):

    pass

  # ────────────────────────────────────────────────────────────────────────
  def rich(self, title=True):
    
    s = ''

    if title:
      s += '[b i #abcdef]fields[/]\n'
    
    s += '─── [i]no field[/]'

    return s

  # ────────────────────────────────────────────────────────────────────────
  def display(self):
    
    print(Panel(self.rich(title=False), title='fields'))
