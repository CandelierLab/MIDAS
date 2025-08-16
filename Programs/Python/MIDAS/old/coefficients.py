'''
Smart coefficients management
'''

import numpy as np
from MIDAS.enums import *
from Programs.Python.MIDAS.ANN.perceptron.canva import PolarGrid

class Coefficients:

  def __init__(self, engine, i, C):

    # --- Relevant properties ----------------------------------------------

    self.engine = engine
    self.i = i

    self.nO = len(self.engine.outputs)
    self.nG = self.engine.groups.N

    if self.engine.inputs[self.i].grid is None:
      self.nCpO = 1

    else:
      grid = self.engine.inputs[self.i].grid
      self.nCpO = self.nG*grid.nZ  
      self.nSa = grid.nSa
      self.nSb = grid.nSb
      
    # Number of coefficients
    self.nC = self.nO*self.nCpO

    # --- Coefficients -----------------------------------------------------

    match type(C).__name__:
      case 'CoeffSet':
        match C:

          case CoeffSet.IGNORE: self.C = np.zeros(self.nC)

      case 'ndarray': self.C = C
      case 'list' | 'tuple': self.C = np.array(C)
      case _: self.C = np.array([C])

  def to_weights(self):
    '''
    Export the coeffificients to an array of weights
    '''

    match self.engine.inputs[self.i].perception:

      case Perception.PRESENCE: 

        if self.engine.inputs[self.i].grid is None:
          return self.C

        W = []

        for i, Out in enumerate(self.engine.outputs):
          for j in range(self.nCpO):

            k = self.nCpO*i + j

            match Out.action:

              case Action.SPEED_MODULATION:
                W.append(self.C[k] if ((j+self.nSa/4) % self.nSa)<self.nSa/2 else -self.C[k])

              case Action.REORIENTATION:
                W.append(self.C[k] if (j % self.nSa)<self.nSa/2 else -self.C[k])

        return np.array(W)
      
      case Perception.ORIENTATION: 

        return self.C
      
      case _:

        return self.C