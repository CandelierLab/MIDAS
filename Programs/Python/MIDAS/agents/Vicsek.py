'''
Vicsek agents
'''

import project
from agents.agent import *

from AE.Display.Animation.Items_2d import *

# === Vicsek agents ========================================================

class Vicsek(agent):
  '''
  Vicsek agent
  '''

  def __init__(self, r, **kwargs):    

    # Generic agent constructor
    super().__init__(**kwargs)

    self.r = r
    
  def add_information(self):

    s = 'r = {:.03f}<br>'.format(self.r)
    s += '<i>&sigma;</i><sub>noise</sub> = {:.03f}<br>'.format(self.noise)
    
    # Add info
    self.engine.window.information.add(text, 'Info',
        stack = True,
        string = s,
        color = 'white',
        fontsize = 12,
      )

  def update(self, iteration, F):
    '''
    Update angles and move

    To compute the average orientation we use the same definition as in:
    | Novel Type of Phase Transition in a System of Self-Driven Particles
    | T. Vicsek, A. Czirók, E. Ben-Jacob, I. Cohen, O. Shochet
    |  https://doi.org/10.1103/PhysRevLett.75.1226
    '''

    mean_orientation = lambda x : np.arctan(np.mean(np.sin(x))/np.mean(np.cos(x))) if x.size else 0

    # An alternate way, managing the empty case better
    mean_orientation_2 = lambda x : np.angle(np.exp(1j*x).sum())

    # # === No reorientation =================================================
    # '''
    # Slightly faster, but not compatible with the original definition, since the empty case leads to an undetermination.
    # '''

    # # Update perception
    # self.perceive(F, r=self.r, reorient=False, include_self=True)
    
    # # Update angle
    # self.a = mean_orientation_2(self.alpha)

    # === With reorientation ===============================================

    # Update perception
    self.perceive(F, r=self.r, reorient=True, include_self=True)
    
    # Update angle
    self.a += mean_orientation_2(self.alpha)

    # Add angular noise and move
    self.move()

    return self
