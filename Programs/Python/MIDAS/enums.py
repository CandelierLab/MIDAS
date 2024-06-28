'''
Enumerations
'''

from enum import Enum, IntEnum
import numpy as np

# === Arena geometries =====================================================

class Arena(IntEnum):
  CIRCULAR = 0
  RECTANGULAR = 1

# === Agent types ==========================================================

class Agent(IntEnum):
  FIXED = 0
  RIPO = 1

# === Coefficients sets ====================================================

class CoeffSet(IntEnum):
  IGNORE = 0
  ATTRACTION = 1

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
  SAME_RADIUS = 201       # Normalization over zones with the same radius
  SAME_SLICE = 202        # Normalization over zones within the same angular slice
  SAME_GROUP = 203        # Normalization over all zones among the same group
  ALL = 204               # Normalization over all zones of all groups

# === Actions ==============================================================

class Action(IntEnum):
  SPEED_MODULATION = 0            # Speed modulation
  REORIENTATION = 1               # ┐ Reorientation (transverse - for easy use in 2D)
  REORIENTATION_TRANSVERSE = 1    # │ Reorientation (transverse plane)
  REORIENTATION_AXIAL = 1         # ┘ 
  REORIENTATION_LONGITUDINAL = 2  # ┐ Reorientation (longitudinal plane)
  REORIENTATION_SAGITTAL = 2      # ┘ 
  REORIENTATION_FRONTAL = 3       # ┐ Reorientation (frontal plane)
  REORIENTATION_CORONAL = 3       # ┘

# === Activation fuctions ==================================================

class Activation(IntEnum):
  IDENTITY = 0            # No activation
  HSM_POSITIVE = 1        # Half-softmax with output in [0,1]
  HSM_CENTERED = 2        # Half-softmax with output in [-1,1]

# === Default values =======================================================

class Default(Enum):

  # Agent parameters
  vmin = 0.               # Minimal speed
  vmax = 0.01             # Maximal speed
  rmax = -1               # Maximal radius (negative value means no rmax)
  dv_scale = 1            # Speed modulation scale
  da_scale = np.pi/2      # Reorientation scale
  db_scale = np.pi/2      # Reorientation scale
  dc_scale = np.pi/2      # Reorientation scale
  vnoise = 0              # Speed noise
  anoise = 0              # Reorientation noise
  bnoise = 0              # Reorientation noise
  cnoise = 0              # Reorientation noise

# === Verbose level ========================================================

class Verbose(IntEnum):
  NONE = 0
  NORMAL = 1
  HIGH = 2