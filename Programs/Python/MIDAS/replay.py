
import numpy as np

from Animation.Window import Window
import MIDAS.animation

from MIDAS.enums import *
from MIDAS.storage import Storage
from MIDAS.engine import Geometry,Agents, Groups

# === REPLAY ===============================================================

class Replay():

  def __init__(self, db_file, style='dark'):

    # --- Storage

    self.storage = Storage(db_file)

    # --- Simulation parameters

    # Get parameters
    param = {}
    for res in self.storage.db_curs.execute("SELECT * FROM Parameters").fetchall():
      param[res[0]] = res[1]

    # Simulation duration
    self.duration = self.storage.db_curs.execute("SELECT MAX(step) FROM Kinematics").fetchone()[0]+1


    # --- Geometry ---------------------------------------------------------

    # Definitions
    self.dimension = param['dimension']

    match self.dimension:

      case 1:
        arena_shape = [param['arena_X']]
        periodic = [param['periodic_X']]

      case 2:
        arena_shape = [param['arena_X'], param['arena_Y']]
        periodic = [param['periodic_X'], param['periodic_Y']]

      case 3:
        arena_shape = [param['arena_X'], param['arena_Y'], param['arena_Z']]
        periodic = [param['periodic_X'], param['periodic_Y'], param['periodic_Z']]
    
    self.geom = Geometry(param['dimension'],
                         arena = param['arena'],
                         shape = arena_shape,
                         periodic = periodic)

    # --- Agents -----------------------------------------------------------

    # Definitons
    self.agents = Agents(self.dimension)
    self.agents.group = np.array(self.storage.db_curs.execute("SELECT gid FROM Agents").fetchall())   
    self.agents.N = self.agents.group.size
      
    # Initial position and velocities
    self.step(0)

    # --- Groups -----------------------------------------------------------

    self.groups = Groups()

    for res in self.storage.db_curs.execute("SELECT * FROM Groups").fetchall():
      self.groups.N += 1
      self.groups.atype.append(res[1])
      self.groups.names.append(res[2])

    # --- Animation --------------------------------------------------------

    # animation_type = Animation.AGENTS
    animation_type = Animation.FIELD_DENSITY

    self.window = Window('MIDAS (replay)', style=style)
    self.window.step_max = self.duration-1

    match self.dimension:
      case 1:
        pass
      case 2:
        match animation_type:

          case Animation.AGENTS:
            self.animation = MIDAS.animation.Agents_2d(self)

          case Animation.FIELD_DENSITY:
            self.animation = MIDAS.animation.Field(self)

      case 3:
        pass
    
    self.window.add(self.animation)

    # Backward animation
    self.window.allow_backward = True
    self.window.allow_negative_time = False

  def step(self, i):
    '''
    Fetch kinematic parameters at every step of the siimulation
    '''

    match self.dimension:
      case 1: sql = f'SELECT x, r FROM Kinematics WHERE step={i}'
      case 2: sql = f'SELECT x, y, r, a FROM Kinematics WHERE step={i}'
      case 3: sql = f'SELECT x, y, z, r, a, b FROM Kinematics WHERE step={i}'

    res = np.array(self.storage.db_curs.execute(sql).fetchall())

    match self.dimension:
      case 1:
        self.agents.pos = res[:,0]
        self.agents.vel = res[:,1]
      case 2:
        self.agents.pos = res[:,0:2]
        self.agents.vel = res[:,2:4]
      case 3:
        self.agents.pos = res[:,0:3]
        self.agents.vel = res[:,3:6]

  def run(self):
    '''
    Run replay
    '''

    self.animation.initialize()
    self.window.show()