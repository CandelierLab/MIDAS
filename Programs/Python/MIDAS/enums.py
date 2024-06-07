'''
Enumerations
'''

from enum import Enum
import numpy as np

# === Arena geometries =====================================================

class Arena(Enum):
  CIRCULAR = np.int32(0)
  RECTANGULAR = np.int32(1)

# === Agent types ==========================================================

class Agent(Enum):
  FIXED = 0
  BLIND = 1
  RIPO = 2
  RINNO = 3

# === Input types ==========================================================

class Input(Enum):
  NOISE = 0         # Gaussian noise
  WALLS = 1         # Walls
  PRESENCE = 2      # Presence (count agents)
  ORIENTATION = 3   # Average orientation 
  FIELD = 4         # Field

# === Normalization ========================================================

class Normalization(Enum):
  NONE = 0          # No normalization
  SAME_RADIUS = 1   # Normalization over the sectors with the same radius
  ALL = 2           # Normalization over all sectors
