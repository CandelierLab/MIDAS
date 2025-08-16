'''
MIDAS core: initial conditions
'''

import numpy as np
from rich import print
from rich.panel import Panel

class initial_conditions:

  # ────────────────────────────────────────────────────────────────────────
  def __init__(self, group):

    # Define group
    self.group = group

    # Default values
    self.position = None
    self.velocity = None
    self.orientation = None
    self.speed = 0.01

  # ────────────────────────────────────────────────────────────────────────
  def rich(self, title=True):

    s = ''

    if title:
      s += f'[b i #abcdef]Initial conditions[/]\n'
    
    s += f'position: {self.position}\n'
    if self.velocity is not None:
      s += f'position: {self.velocity}\n'
    else:
      s += f'orientation: {self.orientation}\n'
      s += f'speed: {self.speed}\n'

    return s

  # ────────────────────────────────────────────────────────────────────────
  def display(self):
    
    print(Panel(self.rich(title=False), title='Initial conditions'))
