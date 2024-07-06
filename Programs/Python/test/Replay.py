'''
Build an animation based on a database
'''

import os

from MIDAS.replay import Replay
from MIDAS.enums import *

os.system('clear')

R = Replay('/home/raphael/Science/Projects/CM/MovingAgents/Data/RIPO/test.db')

# === Visualization ========================================================

R.setup_animation(agents=AnimAgents.SUBSET_100, field=AnimField.DENSITY)
R.animation.trace_duration = 10
# E.animation.group_options['agents']['cmap'] = 'hsv'
R.animation.field_options['range'] = [0, 1]


# # === Simulation ===========================================================

R.window.autoplay = False
# # # E.window.movieFile = movieDir + 'Donut.mp4'

R.run()