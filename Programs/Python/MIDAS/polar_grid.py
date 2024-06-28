'''
Polar grid
'''

import numpy as np

class PolarGrid:
  '''
  Polar grid
  '''

  def __init__(self, rZ=[], rmax=-1, nSa=1, nSb=1):

    self.rZ = np.array(rZ)
    self.nR = self.rZ.size + 1
    self.rmax = rmax

    self.nSa = nSa
    self.nSb = nSb    

    self.nZ = self.nR*self.nSa*self.nSb