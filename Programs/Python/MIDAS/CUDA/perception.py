'''
Perception function
'''

import math, cmath
import numba as nb
from numba import cuda
from MIDAS.enums import *

@cuda.jit(device=True)
def perceive(vIn, p, numbers, geometry, agents, perceptions, custom, z0, v, a, z, alpha, visible, m_nI):

  dim = numbers[0]
  nG = numbers[2]
  nR = numbers[3]
  nSa = numbers[4]
  nSb = numbers[5]

  match perceptions[p,0]:

    case Perception.PRESENCE.value | Perception.ORIENTATION.value:

      if perceptions[p,0]==Perception.ORIENTATION.value:
        Cbuffer = cuda.local.array(m_nI, nb.complex64)

      for j in range(agents.shape[0]):

        # Skip self-perception
        if not visible[j]: continue

        # Perception rmax
        rmax = perceptions[p,3]
        if rmax>0 and abs(z[j])>rmax: continue

        # --- Indices (grid, coefficient)

        # Radial index
        ri = 0
        for k in range(nR):
          ri = k
          if abs(z[j])<perceptions[p, dim+3+k]: break
          
        # Angular index
        ai = int((cmath.phase(z[j]) % (2*math.pi))/2/math.pi*nSa) if dim>1 else 0
        bi = 0 # if dim>2 else 0  # TODO: 3D

        # Grid index
        ig = (ri*nSa + ai)*nSb + bi

        # Coefficient index
        ic = int(agents[j,0]*nR*nSa*nSb + ig)

        # --- Inputs

        match perceptions[p,0]:

          case Perception.PRESENCE.value:
            vIn[ic] += 1

          case Perception.ORIENTATION.value:
            Cbuffer[ic] += cmath.rect(1., alpha[j])

      # --- Post-process

      match perceptions[p,0]:

        case Perception.ORIENTATION.value:

          for ic in range(nG*nR*nSb*nSa):
            vIn[ic] = cmath.phase(Cbuffer[ic])

  return vIn