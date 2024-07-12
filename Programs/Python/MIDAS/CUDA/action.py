'''
Action function
'''

import math, cmath
import numba as nb
from numba import cuda
from MIDAS.enums import *


@cuda.jit(device=True, cache=True)
def update_velocities(V, vOut, param, properties, rng):

  # --- Definitions

  geometry = param[i_GEOMETRY]
  agents = param[i_AGENTS]
  groups = param[i_GROUPS]
  agent = param[i_AGENT]
  actions = param[i_ACTIONS]

  dim = geometry[0]
  i = int(agent[0])
  gid = int(agent[1])
  nP = int(groups[gid,1])
  nO = groups[gid,2]

  # Velocity limits
  vmin = agents[i,1]
  vmax = agents[i,2]

  dv_scale = agents[i,4]

  match dim:

    case 1:
      vnoise = agents[i,5]

    case 2:
      da_scale = agents[i,5]
      vnoise = agents[i,6]
      anoise = agents[i,7]

    case 3:
      pass
      # da_scale = agents[i,5]
      # db_scale = agents[i,6]
      # dc_scale = agents[i,7]
      # vnoise = agents[i,8]
      # anoise = agents[i,9]
      # bnoise = agents[i,10]
      # cnoise = agents[i,11]

  for oid in range(nO):

    aid = int(groups[gid, int(nP + 3 + oid)])
    otype = actions[aid,0]

    match otype:

      case Action.SPEED_MODULATION.value:
        V[0] += dv_scale*vOut[oid]

      case Action.REORIENTATION.value: 
        V[1] += da_scale*vOut[oid]

  # --- Noise ----------------------------------------------------

  match dim:

    case 2:

      # Speed noise
      if vnoise:
        V[0] += vnoise*cuda.random.xoroshiro128p_normal_float32(rng, i)

      # Speed limits
      if V[0] < vmin: V[0] = vmin
      elif V[0] > vmax: V[0] = vmax

      # Angular noise
      if anoise:
        V[1] += anoise*cuda.random.xoroshiro128p_normal_float32(rng, i)

  return V