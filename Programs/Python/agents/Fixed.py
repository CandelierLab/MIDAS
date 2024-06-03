'''
Fixed agents
'''

import project
from agents.agent import *

# === Blind agents =========================================================

class Fixed(agent):
  '''
  Fixed points
  '''

  def __init__(self, **kwargs):    

    # Generic agent constructor
    super().__init__(**kwargs)
    
  def add_insight(self):
    pass

  
