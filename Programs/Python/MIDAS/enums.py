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
  RIPO = 1

# === Perception functions =================================================

class Perception(IntEnum):

  # Agent-dependent
  PRESENCE = 100          # Presence (count agents)
  ORIENTATION = 101       # Average orientation 
  FIELD_AGENTS = 102      # Field
  CUSTOM_AGENTS = 103     # Custom input

  # Non agent-dependent
  NOISE = 104             # Gaussian noise
  WALLS = 105             # Walls
  FIELD = 106             # Field
  CUSTOM = 107            # Custom input

# === Normalization ========================================================

class Normalization(IntEnum):
  NONE = 200              # No normalization
  SAME_RADIUS = 201       # Normalization over sectors with the same radius
  SAME_SLICE = 202        # Normalization over sectors within the same angular slice
  ALL = 203               # Normalization over all sectors

# === Outputs ==============================================================

class Output(IntEnum):
  REORIENTATION = 0       # Reorientation
  SPEED_MODULATION = 1    # Speed modulation

# === Activation fuctions ==================================================

class Activation(IntEnum):
  ANGLE = 0               # Angular activation
  SPEED = 1               # Speed activation
