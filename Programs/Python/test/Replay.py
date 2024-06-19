'''
Build an animation based on a database
'''

import os

from MIDAS.replay import Replay

os.system('clear')

R = Replay('/home/raphael/Science/Projects/CM/MovingAgents/Data/RIPO/test.db')

# === Visualization ========================================================

R.animation.options['agents']['cmap'] = 'hsv'

# # --- Information

# # E.animation.add_info_weights()
# # E.animation.add_info()

# # --- Traces
# # E.animation.trace_duration = 10

# # === Simulation ===========================================================

R.window.autoplay = False
# # # E.window.movieFile = movieDir + 'Donut.mp4'

R.run()