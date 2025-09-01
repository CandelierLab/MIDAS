import os
import time
import numpy as np

import MIDAS

# from MIDAS.enums import *
# from MIDAS.polar_grid import PolarGrid
# from MIDAS.engine import Engine

os.system('clear')

# ═══ Parameters ═══════════════════════════════════════════════════════════

dataDir = '/home/raphael/Bureau/data/'
movieDir = '/home/raphael/Bureau/'

# ═══ Engine ═══════════════════════════════════════════════════════════════

# E = MIDAS.engine(type=MIDAS.ARENA.CIRCULAR)
E = MIDAS.engine()

# Number of steps
E.steps = None

# ═══ Agents ═══════════════════════════════════════════════════════════════

# Group of agents
gP = MIDAS.agents.perceptron(E, 1000, name='SSP')

# Initial conditions
gP.initial.speed = 0.01

# ─── I/O ───────────────────────────────────────

# Spatial canva
canva = MIDAS.ANN.canva.spatial()
canva.radii = [0.05]
canva.number_of_azimuths = 4

# ─── Inputs

gP.input(MIDAS.PERCEPTION.DENSITY,
         perceived = gP, 
         canva = canva,
         coefficients = np.array([1, 1, 1, 1])*1, 
         normalization = MIDAS.NORMALIZATION.SAME_GROUP)

# # ─── Outputs 

# gSSP.output(MIDAS.ACTION.REORIENTATION, activation = MIDAS.ACTIVATION.HSM_CENTERED)

# ═══ Storage ══════════════════════════════════════════════════════════════

# E.storage = MIDAS.storage(dataDir + 'test.db')
# E.storage.commit_frequency = MIDAS.COMMIT.EVERY_1_STEP

# ═══ Visualization ════════════════════════════════════════════════════════

E.animation = MIDAS.animation(agents = MIDAS.ANIMATION_AGENTS.SUBSET_10,
                              field = MIDAS.ANIMATION_FIELD.DENSITY)

E.animation.window.information.display(True)
E.close_finished_animation = False

# # # --- Agent display options

# # E.animation.trace_duration = 10
# # # E.animation.group_options['agents']['cmap'] = 'hsv'

# # # --- Field display options

E.animation.field_options['range'] = [0, 1/10]

# # # --- Grid

# # # E.animation.gridsize = 0.25

# E.animation.window.movieFile = movieDir + 'test.mp4'

# E.animation.window.autoplay = False

# ═══ Simulation ═══════════════════════════════════════════════════════════

E.run()
