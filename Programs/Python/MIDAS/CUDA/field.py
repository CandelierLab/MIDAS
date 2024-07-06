'''
Field function
'''

import numba as nb
from numba import cuda

@cuda.jit
def update(fields, pos, vel):

  # for i in range(fields.shape[0]):
  #   for j in range(fields.shape[1]):
  #     fields[i,j,0] += 0.1

  pass