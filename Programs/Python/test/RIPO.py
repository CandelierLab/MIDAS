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

# Number of steps
E.steps = 10

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
               'coefficients': [1, 1, -1, -1]})

# Outputs 
outputs = {Output.REORIENTATION: Activation.ANGLE}

# Initial conditions
N = 100
IC = {'position': None,
      'orientation': None,
      'speed': 0.01} 

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

# === Storage ==============================================================

E.setup_storage('/home/raphael/Science/Projects/CM/MovingAgents/Data/RIPO/test.db')

# === Visualization ========================================================

E.setup_animation()
E.animation.options['agents']['cmap'] = 'hsv'

# --- Information

# E.animation.add_info_weights()
# E.animation.add_info()

# --- Traces
# E.animation.trace_duration = 10

# === Simulation ===========================================================

# E.window.autoplay = False
# # E.window.movieFile = movieDir + 'Donut.mp4'

E.run()