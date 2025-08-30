'''
Enumerations
'''

from enum import Enum, IntEnum
import numpy as np

# ═══ Arena geometry ═══════════════════════════════════════════════════════

class ARENA(IntEnum):
  
  CIRCULAR = 0
  RECTANGULAR = 1

# ═══ Agent types ══════════════════════════════════════════════════════════

# class Agent(IntEnum):

#   FIXED = 0
#   SSP = 1
#   TSP = 2

# ═══ Perception ═══════════════════════════════════════════════════════════

class PERCEPTION(IntEnum):

  # Base perceptions
  DENSITY = 1             # Presence (count agents)
  ORIENTATION = 2         # Average orientation 

  # Fields
  FIELD = 0               # First field (alias)
  FIELD_0 = 0             # First field
  FIELD_1 = -1            # Second field
  FIELD_2 = -2            # Third field

  # Misc
  WALL = 100              # Walls (not implemented)

class NORMALIZATION(IntEnum):

  NONE = 0                # No normalization
  SAME_RADIUS = 1         # Normalization over zones with the same radius
  SAME_SLICE = 2          # Normalization over zones within the same angular slice
  SAME_GROUP = 3          # Normalization over all zones among the same group
  ALL = 4                 # Normalization over all zones of all groups

class GROUP(IntEnum):

  ALL = 0                 # Perceive agents from all the groups
  SELF = 1                # Perceive only agents from the same group

# ═══ Animation ════════════════════════════════════════════════════════════

class ANIMATION_AGENTS(IntEnum):

  NONE = 0
  SUBSET_1 = 1
  SUBSET_10 = 2
  SUBSET_100 = 3
  ALL = 100

class ANIMATION_FIELD(IntEnum):

  NONE = -1
  DENSITY = -2
  POLARITY = -3

# ═══ Storage ══════════════════════════════════════════════════════════════

class COMMIT(IntEnum):

  AT_THE_END = 0
  EVERY_1_STEP = 1
  

# # === Coefficients sets ====================================================

# class CoeffSet(IntEnum):

#   IGNORE = 0
#   #  PRESENCE_ATTRACTION = 1     To implement

# # === Actions ==============================================================

# class Action(IntEnum):

#   SPEED_MODULATION = 0            # Speed modulation
#   REORIENTATION = 1               # ┐ Reorientation (transverse - for easy use in 2D)
#   REORIENTATION_TRANSVERSE = 1    # │ Reorientation (transverse plane)
#   REORIENTATION_AXIAL = 1         # ┘ 
#   REORIENTATION_LONGITUDINAL = 2  # ┐ Reorientation (longitudinal plane)
#   REORIENTATION_SAGITTAL = 2      # ┘ 
#   REORIENTATION_FRONTAL = 3       # ┐ Reorientation (frontal plane)
#   REORIENTATION_CORONAL = 3       # ┘

# # === Activation fuctions ==================================================

# class Activation(IntEnum):

#   IDENTITY = 0            # No activation
#   HSM_POSITIVE = 1        # Half-softmax with output in [0,1]
#   HSM_CENTERED = 2        # Half-softmax with output in [-1,1]

# # === Default values =======================================================

# class Default(Enum):

#   # Agent parameters
#   vmin = 0.               # Minimal speed
#   vmax = 0.01             # Maximal speed
#   rmax = -1               # Maximal radius (negative value means no rmax)
#   dv_scale = 1            # Speed modulation scale
#   da_scale = np.pi/2      # Reorientation scale
#   db_scale = np.pi/2      # Reorientation scale
#   dc_scale = np.pi/2      # Reorientation scale
#   vnoise = 0              # Speed noise
#   anoise = 0              # Reorientation noise
#   bnoise = 0              # Reorientation noise
#   cnoise = 0              # Reorientation noise

# # === Verbose level ========================================================

# class Verbose(IntEnum):

#   NONE = 0
#   NORMAL = 1
#   HIGH = 2

# # === Animations ===========================================================

# class AnimAgents(IntEnum):

#   NONE = 0
#   ALL = 1
#   SUBSET_10 = 2
#   SUBSET_100 = 3

# class AnimField(IntEnum):

#   NONE = -1
#   DENSITY = -2
#   POLARITY = -3

# # === Indices ==============================================================
# '''
# Do not change these values without updating the CUDA kernel
# '''

# i_GEOMETRY = 0
# i_GROUPS = 1
# i_AGENTS = 2
# i_PERCEPTIONS = 3
# i_ACTIONS = 4
# i_AGENT = 5
# i_AGENTS_POSITIONS = 6
# i_AGENTS_ORIENTATIONS = 7
# i_AGENTS_VISIBILITY = 8
# i_FIELDS = 9
# i_CUSTOM = 10

# ip_MNIPP = 0
# ip_NO = 1
# ip_NG = 2
# ip_NR = 3
# ip_NSA = 4
# ip_NSB = 5
# ip_FIELD_OFFSET = 6