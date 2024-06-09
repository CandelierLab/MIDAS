'''
Enumerations
'''

from enum import IntEnum
import numpy as np

# === Arena geometries =====================================================

class Arena(IntEnum):
  CIRCULAR = np.int32(0)
  RECTANGULAR = np.int32(1)

# === Agent types ==========================================================

class Agent(IntEnum):
  FIXED = 0
  BLIND = 1
  RIPO = 2
  RINNO = 3

# === Radial input types ===================================================

class RInput(IntEnum):
  NOISE = 0         # Gaussian noise
  WALLS = 1         # Walls
  PRESENCE = 2      # Presence (count agents)
  ORIENTATION = 3   # Average orientation 
  FIELD = 4         # Field
  CUSTOM = 5        # Custom input

# === Normalization ========================================================

class Normalization(IntEnum):
  NONE = 0          # No normalization
  SAME_RADIUS = 1   # Normalization over sectors with the same radius
  ALL = 2           # Normalization over all sectors
