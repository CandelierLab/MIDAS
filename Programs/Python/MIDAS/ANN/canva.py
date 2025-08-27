'''
ANN input canvas
'''

import numpy as np
from rich import print
from rich.panel import Panel

# ══════════════════════════════════════════════════════════════════════════
#                               spatial canva
# ══════════════════════════════════════════════════════════════════════════

class spatial:

  # ────────────────────────────────────────────────────────────────────────
  def __init__(self, dimension=2, radii=[], number_of_azimuths=1, number_of_altitudes=1):

    # Dimension
    self.dimension = dimension

    # Radii
    self.R = np.array(radii)
    
    # Angular slices
    self.nAz = number_of_azimuths if self.dimension>1 else 1
    self.nAl = number_of_altitudes if self.dimension>2 else 1

  @property
  def radii(self): return self.R

  @radii.setter
  def radii(self, R): self.R = np.array(R)

  @property
  def number_of_azimuths(self): return self.nAz

  @number_of_azimuths.setter
  def number_of_azimuths(self, nAz): self.nAz = nAz

  @property
  def number_of_altitudes(self): return self.nAl

  @number_of_altitudes.setter
  def number_of_altitudes(self, nAl): self.nAl = nAl

  @property
  def nR(self): return self.R.size

  @property
  def rmax(self): return self.R[-1] if self.R.size else None

  @property
  def nIn(self): return self.nR*self.nAz*self.nAl

  # ────────────────────────────────────────────────────────────────────────
  def rich(self, title=True):
    
    s = ''

    if title:
      s += f'[b i #abcdef]{self.dimension}d spatial canva[/]\n'

    s += f'radii: {self.R}\n'
    s += f'number of azimuths: {self.nAz}\n'
    s += f'number of altitudes: {self.nAl}\n'
    s += f'number of inputs: {self.nIn}'

    return s

  # ────────────────────────────────────────────────────────────────────────
  def display(self): 

    print(Panel(self.rich(title=False), title=f'{self.dimension}d spatial canva'))
