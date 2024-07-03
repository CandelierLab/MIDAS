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
  #  PRESENCE_ATTRACTION = 1     To implement

# === Perception functions =================================================

class Perception(IntEnum):

  # Agent-dependent
  PRESENCE = 0            # Presence (count agents)
  ORIENTATION = 1         # Average orientation 
  FIELD_AGENTS = 2        # Field
  CUSTOM_AGENTS = 3       # Custom input

  # Non agent-dependent
  NOISE = 4               # Gaussian noise
  WALLS = 5               # Walls
  FIELD = 6               # Field
  CUSTOM = 7              # Custom input

# === Normalization ========================================================

class Normalization(IntEnum):

  NONE = 0                # No normalization
  SAME_RADIUS = 1         # Normalization over zones with the same radius
  SAME_SLICE = 2          # Normalization over zones within the same angular slice
  SAME_GROUP = 3          # Normalization over all zones among the same group
  ALL = 4                 # Normalization over all zones of all groups

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

# === Animation type ========================================================

class Animation(IntEnum):

  AGENTS = 0
  FIELD_DENSITY = 1

  CUSTOM = 100
