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
E = MIDAS.engine()

# Number of steps
E.steps = 100

# ═══ Agents ═══════════════════════════════════════════════════════════════

# Group of agents
gP = MIDAS.agents.perceptron(E, 100000, name='SSP')

# Initial conditions
gP.initial.speed = 0.01

# ─── I/O ───────────────────────────────────────

# Spatial canva
canva = MIDAS.ANN.canva.spatial()
canva.radii = [0.5]
canva.number_of_azimuths = 4

# ─── Inputs

gP.input(MIDAS.PERCEPTION.DENSITY,
         perceived = gP, 
         canva = canva,
         coefficients = np.array([1, 1, -1, -1])*1, 
         normalization = MIDAS.NORMALIZATION.SAME_GROUP)

# # ─── Outputs 

# gSSP.output(MIDAS.ACTION.REORIENTATION, activation = MIDAS.ACTIVATION.HSM_CENTERED)

# ═══ Visualization ════════════════════════════════════════════════════════

E.animation = MIDAS.animation(agents=MIDAS.ANIMATION_AGENTS.SUBSET_100)

E.animation.window.information.display(True)
E.animation.window.autoplay = False
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

# ═══ Simulation ═══════════════════════════════════════════════════════════

# # E.window.movieFile = movieDir + 'test.mp4'

# # E.window.autoplay = False
E.run()

# E.display()