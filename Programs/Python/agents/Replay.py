'''
Replay agents
'''

import project
from agents.agent import *

# === Replay agents =========================================================

class Replay(agent):
  '''
  Replay agent, i.e. agent from a former run
  '''

  def __init__(self, name, dataFile, box, idx):
    
    self.name = name
    self.idx = idx
    self.dataFile = dataFile
    self.box = box
    
    # Initial positions
    self.x = float(self.dataFile.pos[0,self.idx,0])
    self.y = float(self.dataFile.pos[0,self.idx,1])
    self.a = float(self.dataFile.pos[0,self.idx,2])

    # Density
    self.density = {'pos':[], 'ang':[]}
    self.kde_sigma = {'pos':None, 'ang':None}

    # Blind list
    self.blindlist = None
    
  def update(self, i, F):
    '''
    Update position and angles
    '''

    # Update perception
    I = self.perceive(F, reorient=False)

    # Initial positions
    self.x = float(self.dataFile.pos[i,self.idx,0])
    self.y = float(self.dataFile.pos[i,self.idx,1])
    self.a = float(self.dataFile.pos[i,self.idx,2])
