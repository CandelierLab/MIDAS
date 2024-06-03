'''
NEAT agents
'''

import project
from agents.agent import *

# === Blind agents =========================================================

class NEAT(agent):
  '''
  NEAT ANN-driven agent
  '''

  def __init__(self, v, Nslices, net=None, velocity_modulation=False, sigma=0, box=1, initial_position=None):
    super().__init__(v, sigma, box, initial_position)
    self.v0 = v
    self.net = net
    self.Nslices = Nslices
    self.vmod = velocity_modulation

  def update(self, i, F):
    '''
    Update angles and move
    '''

    # Update perception
    self.perceive(i, F)

    # Values
    v = np.zeros(self.Nslices)
    if len(self.rho):
      # rho1 = np.min(self.rho)
      for k in range(self.Nslices):

        # Indices
        I = np.where((self.theta>=2*k*np.pi/self.Nslices) & (self.theta<=2*(k+1)*np.pi/self.Nslices))

        # Values      
        v[k] = np.sum((1+self.rho[I])**-1)/len(self.rho)
        # v[k] = np.sum(rho1/self.rho[I])

    else:
      v = [0]*self.Nslices

    # Renormalization
    v = v/np.sum(v)

    # Network activation
    V = self.net.activate(v)
    
    # Update angle
    self.a += V[0]*np.pi/2
    
    # Update speed (if modulations enabled)
    if self.vmod:
      self.v = self.v0*(1+V[1])

    # Add angular noise and move    
    self.move()
