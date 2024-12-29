'''
Perception function
'''

import math, cmath
import numba as nb
from numba import cuda
from MIDAS.enums import *

@cuda.jit(device=True, cache=False)
def perceive(pIn, properties, rng, p, param, pparam):
  '''
  Perception function

  The purpose of this function is to determine all the perception inputs
  sensed by the agents. At each time step it is called once for each set
  of perception input.
  __________________________________________________________________________
  Args:
    pIn (array):
      A CUDA local array (float32) of size m_nIpp, i.e. the maximal number
      of inputs for a given perception set. It has been initialized with 
      zeros in the engine.

    properties (array):
      An array containing something.
      -----------------------------------------------------------
      #!# Currently it is unclear what ; possible to remove ? #!#
      -----------------------------------------------------------

    rng (array):
      The random number generator, created with create_xoroshiro128p_states.
      Do not change or modify.

    p (array):
      Perception index.

    param (tuple):
      A tuple of parameters arrays. These parameters are fixed, they should
      not be altered in the perception function. The tuple element are, in 
      this order:

      ┌───────┬───────────────────────┬─────────────────────────────────────┐
      │ Index │ Index name            │ Description                         │
      ├───────┼───────────────────────┼─────────────────────────────────────┤
      │   0   │ i_GEOMETRY            │ Geometry    ┐                       │
      │   1   │ i_GROUPS              │ Groups      │ Detailed description  │
      │   2   │ i_AGENTS              │ Agents      ├ in engine.py          │
      │   3   │ i_PERCEPTIONS         │ Perceptions │ (class CUDA)          │
      │   4   │ i_ACTIONS             │ Actions     ┘                       │
      ├───────┼───────────────────────┼─────────────────────────────────────┤
      │   5   │ i_AGENT               │ Agent's position and velocity       │
      │       │                       │  agent[0]: Agent index              │
      │       │                       │  agent[1]: Group index              │
      │       │                       │  agent[2]: x-position               │
      │       │                       │  agent[3]: y-position               │
      │       │                       │  agent[4]: x-velocity               │
      │       │                       │  agent[5]: y-velocity               │
      ├───────┼───────────────────────┼─────────────────────────────────────┤
      │   6   │ i_AGENTS_POSITIONS    │
      │   7   │ i_AGENTS_ORIENTATIONS │
      │   8   │ i_AGENTS_VISIBILITY   │
      │   9   │ i_FIELDS              │
      │   10  │ i_CUSTOM              │
      └───────┴───────────────────────┴─────────────────────────────────────┘
      
      param = (geometry, groups, agents, perceptions, actions,
        agent, z, alpha, visible, 
        input_fields, custom_param)
      a

    pparam (array):
      a
  __________________________________________________________________________
  Returns:

    pIn (array): 
      The first parameter.

    properties (array):
      An array containing something.

    rng (array):
      The random number generator.

  '''

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
      # # # for k in range(nSa):
      # # #   pIn[k] = fields[i, k + offset]

  return (pIn, properties, rng)