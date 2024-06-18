import os
import numpy as np
from numba import cuda

from MIDAS.enums import *
from MIDAS.engine import Engine

os.system('clear')

# === Parameters ===========================================================

# Nagents = 100

# movieDir = project.root + '/Movies/TAPAs/'

# === Engine ===============================================================

E = Engine()
# E = Engine(arena=Arena.CIRCULAR)

# # Number of steps
E.steps = None

# Verbose
# E.verbose = False

# === Agents ===============================================================

# --- Blind agents

# E.add_group(Agent.BLIND, Nagents, name='agents', 
#             anoise = 0.1,
#             vnoise = 0.001)

# --- RIPO agents

# Radii of sectors
rS = []

# Number of slices
nSa = 4

# Coefficients
inputs = []
inputs.append({'perception': Perception.PRESENCE, 
               'normalization': Normalization.NONE,
               'coefficients': [1, 1, -1, -1, -1, 1, 1, -1]})

# Outputs 
outputs = {Output.REORIENTATION: Activation.ANGLE,
           Output.SPEED_MODULATION: Activation.SPEED}

# Initial conditions
N = 100
IC = {'position': None,
      'orientation': None,
      'speed': 0.015} 

# IC = {'position': [[0,0], [0.1,0.3]],
#       'orientation': [1.5, 0],
#       'speed': 0.015}
# N = len(IC['position']) 

E.add_group(Agent.RIPO, N, name='agents',
            initial_condition = IC,
            rS = rS, 
            rmax = None,
            nSa = nSa,
            inputs=inputs, outputs=outputs)

# # --- RINNO weights --------------------------------------------------------

# W = RINNO_weights(nS, presence=True, orientation=True)
# # W = TAPA_weights(4, presence=True)

# # --- Presence FB

# # Foreground
# # W.presence.FB[0].foreground = np.array([-1, 1, 1, 1])*1

# # --- Presence LR

# # Foreground
# # W.presence.LR[0].foreground = np.array([-0.4, 0.4, 0.4, 0.4, 0.06, 0.06, -0.06, 0.06, 0.06, 0.06, 0.4, 0.4])*2
# W.presence.LR[0].foreground = [1, 0]

# # Background
# W.presence.LR[0].background = np.array([-1,-1])*rho*1
# # W.presence.LR[0].background = W.presence.LR[0].foreground*rho

# # # --- Orientation

# # W.orientation.LR[0].foreground = 2*np.pi/nS
# # W.orientation.LR[0].background = -np.array([1,1])*1

# # --- RINNOs agents --------------------------------------------------------

# E.agents.add(Nagents, 'RINNO_2d', name='agents', 
#   initial_condition = IC,
#   threshold = rho,
#   weights = W,
#   da_max = np.pi/30,
#   delta = 0,
#   v_max = 0.02,
#   noise = 0.05)

# # Set initial orientation
# # for A in E.agents.list:
# #   A.a = 0

# # === Visualization ========================================================

E.setup_animation()

# --- Agents settings 

# E.animation.options['fixed']['color'] = 'grey'
E.animation.options['agents']['cmap'] = 'hsv'
# E.animation.options['agents']['cmap_on'] = 'index'

# # --- Information

# E.animation.add_info_weights()
# E.animation.add_info()

# # --- Traces
# # E.animation.trace_duration = 10

# # === Simulation ===========================================================

E.window.autoplay = False
# # E.window.movieFile = movieDir + 'Donut.mp4'

E.run()