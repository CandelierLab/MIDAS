import os
import time
import numpy as np

from MIDAS.enums import *
from MIDAS.polar_grid import PolarGrid
from MIDAS.coefficients import Coefficients
from MIDAS.engine import Engine

os.system('clear')

# === Engine ===============================================================

E = Engine(periodic=[False, False])

# Number of steps
E.steps = None

# === Agents ===============================================================

# --- RIPO agents ----------------------------------------------------------

#  --- Field

fbin = [10,10]
field = np.meshgrid(range(fbin[0]), range(fbin[1]))[0]

F = E.add_field(field)

#  --- Inputs

# polar grid
G = PolarGrid(rZ=[], nSa = 4)

in_presence = E.add_input(Perception.PRESENCE,
                          normalization = Normalization.SAME_GROUP,
                          grid = G)

# --- Outputs 

out_da = E.add_output(Action.REORIENTATION,
                      activation = Activation.HSM_CENTERED)

# --- Groups

# Initial conditions
N = 100
IC = {'position': None,
      'orientation': None,
      'speed': 0.01} 

E.add_group(Agent.RIPO, N, name='agents',
            initial_condition = IC,
            inputs=[in_presence], outputs=[out_da])

# --- Coefficients

E.set_weights(in_presence, np.array([1, 1, 1, 1]))

# === Visualization ========================================================

E.setup_animation(agents=AnimAgents.ALL, field=AnimField.DENSITY)
E.animation.trace_duration = 10
E.animation.group_options['agents']['cmap'] = 'hsv'
E.animation.field_options['range'] = [0, 0.01]

# === Simulation ===========================================================

# E.window.autoplay = False

E.run()