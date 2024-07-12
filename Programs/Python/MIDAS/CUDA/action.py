'''
Action function
'''

import math, cmath
import numba as nb
from numba import cuda
from MIDAS.enums import *

@cuda.jit(device=True, cache=True)
def action(pIn, measurements, rng, p, param):

  # --- Definitions

  dim = int(param[i_GEOMETRY][0])

  perceptions = param[i_PERCEPTIONS]
  rmax = perceptions[p,3]

  agents = param[i_AGENTS]
  z = param[i_AGENTS_POSITIONS]
  alpha = param[i_AGENTS_ORIENTATIONS]
  visible = param[i_AGENTS_VISIBILITY]

  nG = param[i_NG]
  nR = param[i_NR]
  nSa = param[i_NSA]
  nSb = param[i_NSB]

  match perceptions[p,0]:

    case Perception.PRESENCE.value | Perception.ORIENTATION.value:

      if perceptions[p,0]==Perception.ORIENTATION.value:
        Cbuffer = cuda.local.array(param[i_MNIPP], nb.complex64)

      for j in range(agents.shape[0]):

        # Skip self-perception
        if not visible[j]: continue

        # Perception rmax
        if rmax>=0 and abs(z[j])>rmax: continue

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
            pIn[ic] += 1

          case Perception.ORIENTATION.value:
            Cbuffer[ic] += cmath.rect(1., alpha[j])

      # --- Post-process

      match perceptions[p,0]:

        case Perception.ORIENTATION.value:

          for ic in range(nG*nR*nSb*nSa):
            pIn[ic] = cmath.phase(Cbuffer[ic])

  return (pIn, measurements, rng)