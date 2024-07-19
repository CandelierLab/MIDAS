'''
Perception function
'''

import math, cmath
import numba as nb
from numba import cuda
from MIDAS.enums import *

@cuda.jit(device=True, cache=True)
def perceive(pIn, properties, rng, p, param, pparam):

  perceptions = param[i_PERCEPTIONS]

  match perceptions[p,0]:

    case Perception.PRESENCE.value | Perception.ORIENTATION.value:

      # --- Definitions

      dim = int(param[i_GEOMETRY][0])

      rmax = perceptions[p,4]

      agents = param[i_AGENTS]
      z = param[i_AGENTS_POSITIONS]
      alpha = param[i_AGENTS_ORIENTATIONS]
      visible = param[i_AGENTS_VISIBILITY]

      nG = pparam[ip_NG]
      nR = pparam[ip_NR]
      nSa = pparam[ip_NSA]
      nSb = pparam[ip_NSB]

      if perceptions[p,0]==Perception.ORIENTATION.value:
        Cbuffer = cuda.local.array(pparam[ip_MNIPP], nb.complex64)

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
          if abs(z[j])<perceptions[p, dim+4+k]: break
          
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

    case Perception.FIELD.value:

      # --- Definitions

      agent = param[i_AGENT]
      fields = param[i_FIELDS]
      nSa = pparam[ip_NSA]
      offset = pparam[ip_FIELD_OFFSET]

      # Parameters
      i = int(agent[0])

      # Transfer inputs
      for k in range(nSa):
        pIn[k] = fields[i, k + offset]

  return (pIn, properties, rng)