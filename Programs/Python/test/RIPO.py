import os
import time
import numpy as np

from MIDAS.enums import *
from MIDAS.engine import Engine

os.system('clear')

# === Parameters ===========================================================

# movieDir = project.root + '/Movies/TAPAs/'

# === Engine ===============================================================

E = Engine()
# E = Engine(arena=Arena.CIRCULAR)

# Number of steps
E.steps = None

# Verbose
# E.verbose.level = Verbose.HIGH

# E.verbose('outside')

# === Agents ===============================================================

# --- Fixed agents ---------------------------------------------------------

# E.add_group(Agent.BLIND, Nagents, name='agents', 
#             anoise = 0.1,
#             vnoise = 0.001)

# --- RIPO agents ----------------------------------------------------------

# Radii of zones
rZones = [0.1]

# Number of angular slices
nAngSlices = 4

#  --- Inputs
in_presence = E.add_input(perception=Perception.PRESENCE,
                          normalization = Normalization.NONE,
                          rZones = rZones,
                          nAngSlices = nAngSlices,
                          coefficients = [1, 1, -1, -1])

# in_orientation = E.add_input({'perception': Perception.ORIENTATION, 
#              'normalization': Normalization.NONE,
#              'coefficients': [1, 1, 1, 1, 0, 0, 0, 0]})

# --- Outputs 
out_da = E.add_output(action = Output.REORIENTATION,
                      activation = Activation.ANGLE)

# Initial conditions
N = 5
IC = {'position': None,
      'orientation': None,
      'speed': 0.01} 

# # IC = {'position': [[0,0], [0.2,0.3]],
# #       'orientation': [1.5, 0],
# #       'speed': 0.015}
# # N = len(IC['position']) 

E.add_group(Agent.RIPO, N, name='agents',
            initial_condition = IC,
            rmax = None,            
            inputs=[in_presence], outputs=[out_da])

aparam, gparam, iparam, oparam = E.define_parameters()

print(iparam)

# === Storage ==============================================================

# E.setup_storage('/home/raphael/Science/Projects/CM/MovingAgents/Data/RIPO/test.db')

# === Visualization ========================================================

# E.setup_animation()
# E.animation.options['agents']['cmap'] = 'hsv'

# # --- Information

# # E.animation.add_info_weights()
# # E.animation.add_info()

# # --- Traces
# # E.animation.trace_duration = 10

# # === Simulation ===========================================================

# # E.window.autoplay = False
# # # E.window.movieFile = movieDir + 'test.mp4'

# E.run()