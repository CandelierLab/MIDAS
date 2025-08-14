'''
Polar grid
'''

import numpy as np
from rich import print
from rich.panel import Panel

class grid:

  # ────────────────────────────────────────────────────────────────────────
  def __init__(self, engine, radii, number_of_azimuths, number_of_altitudes):

    # Dimension
    self.dimension = engine.geometry.dimension

    # Radii
    self.R = np.array(radii)
    self.nR = self.R.size
    self.rmax = self.R[-1] if self.nR else None

    # Angular slices
    self.nAz = number_of_azimuths if self.dimension>1 else 1
    self.nAl = number_of_altitudes if self.dimension>2 else 1

    # Zones
    self.nZ = self.nR*self.nAz*self.nAl

  # ────────────────────────────────────────────────────────────────────────
  def rich(self, title=True):
    
    s = ''

    if title:
      s += f'[b i #abcdef]{self.dimension}d polar grid[/]\n'

    s += f'radii: {self.R}\n'
    s += f'number of azimuths: {self.nAz}\n'
    s += f'number of altitudes: {self.nAl}'

    return s

  # ────────────────────────────────────────────────────────────────────────
  def display(self): 

    print(Panel(self.rich(title=False), title=f'{self.dimension}d polar grid'))
