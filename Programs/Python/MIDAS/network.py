'''
Network functions

TO DO

'''

import math, cmath
import numba as nb
from numba import cuda
from MIDAS.enums import *

class Networks:

  def __init__(self) -> None:
    pass

class Network:

  def __init__(self) -> None:
    pass

############################################################################
# ######################################################################## #
# #                                                                      # #
# #                                                                      # #
# #                               CUDA                                   # #
# #                                                                      # #
# #                                                                      # #
# ######################################################################## #
############################################################################

'''
network is a tuple (param, nodes, edges)
'''

@cuda.jit(device=True, cache=True)
def run(vOut, measurements, vIn, param):

  # --- Definitions

  nI = vIn.size
  nO = vOut.size

  geometry = param[i_GEOMETRY]  
  agent = param[i_AGENT]  
  groups = param[i_GROUPS]
  perceptions = param[i_PERCEPTIONS]

  dim = geometry[0]
  gid = int(agent[1])
  nG = groups.shape[0]
  nP = groups[gid,1]

  k = 0

  for pi in range(nP):

    # Perception index
    p = int(groups[gid, pi+3])

    # Number of inputs
    nR = perceptions[p,2]
    nSa = perceptions[p,4] if dim>1 else 1
    nSb = perceptions[p,5] if dim>2 else 1
    nIpp = nG*nR*nSa*nSb

    for u in range(nO):
      for v in range(nIpp):

        vOut[u] += vIn[k]*perceptions[p, int(dim + nR + 3 + v)]    
        k += 1

  return (vOut, measurements)