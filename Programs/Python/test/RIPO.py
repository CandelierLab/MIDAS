import os
import numpy as np

from MIDAS.engine import Engine

os.system('clear')

# === Parameters ===========================================================

Nagents = 1000

# movieDir = project.root + '/Movies/TAPAs/'

# # === Engine ===============================================================

E = Engine()

# # --- General settings

# # Arena
# E.boxSize = 1
# E.periodic_boundary_condition = True

# # Number of steps
E.steps = 2

# # Verbose
# E.verbose = None

# # === Agents ===============================================================

E.add_group('blind', Nagents, name='agents')

# nS = 4
# rho = 0.2

# IC = {'position': None,
#       'orientation': None,
#       'speed': 0.01}

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

# E.animation.options['agents']['color'] = 'red'
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