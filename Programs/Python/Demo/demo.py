import os
import time
import numpy as np

import MIDAS

# from MIDAS.enums import *
# from MIDAS.polar_grid import PolarGrid
# from MIDAS.engine import Engine

os.system('clear')

# ═══ Parameters ═══════════════════════════════════════════════════════════

# dataDir = '/home/raphael/Science/Projects/CM/MovingAgents/Data/'
# movieDir = '/home/raphael/Science/Projects/CM/MovingAgents/Movies/'

# ═══ Engine ═══════════════════════════════════════════════════════════════

E = MIDAS.engine()

# E = Engine(arena=Arena.CIRCULAR)

# # Number of steps
# E.steps = None

# # === Agents ===============================================================

# ────────────────────────────────────────────────────────────────────────

# # --- RIPO agents ----------------------------------------------------------

# #  --- Inputs

# # polar grid
# G = PolarGrid(rZ=[], nSa=4)

# in_presence = E.add_input(Perception.PRESENCE,
#                           normalization = Normalization.SAME_GROUP,
#                           grid = G)

# in_orientation = E.add_input(Perception.ORIENTATION,
#                              normalization = Normalization.NONE,
#                              grid = G)

# # --- Outputs 

# out_da = E.add_output(Action.REORIENTATION,
#                       activation = Activation.HSM_CENTERED)

# # out_dv = E.add_output(Action.SPEED_MODULATION,
# #                       activation = Activation.HSM_CENTERED)

# # --- Groups

# # Initial conditions
# N = 100
# IC = {'position': None,
#       'orientation': None,
#       'speed': 0.01}

# E.add_group(Agent.RIPO, N, name='agents',
#             initial_condition = IC,
#             inputs=[in_presence, in_orientation], outputs=[out_da])

# # --- Coefficients

# E.set_coefficients(in_presence, np.array([1, 1, 1, 1])*-1)
# E.set_coefficients(in_orientation, np.array([1, 1, 1, 1])*0)

# # === Storage ==============================================================

# # E.setup_storage(dataDir + 'RIPO/test.db')
# # E.storage.db_commit_each_step = True

# # === Visualization ========================================================

# E.setup_animation(agents=AnimAgents.SUBSET_100, field=AnimField.DENSITY)

# # --- Agent display options

# E.animation.trace_duration = 10
# # E.animation.group_options['agents']['cmap'] = 'hsv'

# # --- Field display options

# E.animation.field_options['range'] = [0, 1/N]

# # --- Grid

# E.animation.gridsize = 0.25

# # === Simulation ===========================================================

# # E.window.movieFile = movieDir + 'test.mp4'

# # E.window.autoplay = False
# E.run()