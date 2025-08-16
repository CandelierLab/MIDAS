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

# E = MIDAS.engine(type=MIDAS.ARENA.CIRCULAR)
E = MIDAS.engine(periodic=[False, True])

# Number of steps
E.steps = 100

E.display()

# ═══ Agents ═══════════════════════════════════════════════════════════════

# Group of agents
gSSP = MIDAS.agents.SSP(E, 100, name='SSP')

# Initial conditions
gSSP.initial.speed = 0.01

# ─── I/O ───────────────────────────────────────

# Spatial canva
# gSSP.canva.radii = [0.1]
# gSSP.canva.number_of_azimuths = 4

# ─── Inputs

gSSP.input(MIDAS.PERCEPTION.DENSITY, 
           normalization = MIDAS.NORMALIZATION.SAME_GROUP,
           perceived = gSSP, 
           coefficients = np.array([1, 1, 1, 1])*1)

# # # # ─── Outputs 

# # # gSSP.output(MIDAS.ACTION.REORIENTATION, activation = MIDAS.ACTIVATION.HSM_CENTERED)

gSSP.display()

# gSSP.canva.display()

# ═══ Visualization ════════════════════════════════════════════════════════

# E.animation = MIDAS.animation()

# E.animation.window.autoplay = False
# E.animation.window.show()




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