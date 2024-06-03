'''
Blind agents
'''

import project
from agents.agent import *

# === Blind agents =========================================================

class Blind(agent):
  '''
  Blind agent, i.e. not taking the other agents into account 
  '''

  def __init__(self, **kwargs):    

    # Generic agent constructor
    super().__init__(**kwargs)
    
  def add_information(self):

    s = '<i>&sigma;</i><sub>noise</sub> = {:.03f}<br>'.format(self.noise)
    
    self.engine.animation.add_insight_text('param', s)

  def update(self, iteration, F):
    '''
    Update angles and move
    '''

    # Update perception
    I = self.perceive(F, reorient=False)

    # Add angular noise and move
    self.move()
