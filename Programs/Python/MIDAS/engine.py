from math import *
import numpy as np
import time

import project
from Animation.Window import Window                           # type: ignore

# import animation

# === List of agents =======================================================

# class Agents:
#   '''
#   Collection of agents
#   '''

#   def __init__(self, engine):

#     self.N = 0
#     self.list = []
#     self.groups = {}
#     self.groupnames = []
#     self.groupIndex = []
        
#     # Engine
#     self.engine = engine

#   def __str__(self):
#     s = '--- Agents ---\n'
#     s += 'N: ' + str(self.N)
#     return s

#   def add(self, n, atype, initial_condition, noise=0, damax_range=None, **kwargs):
#     '''
#     Add one or many agents
#     '''

#     # Default name
#     if 'name' not in kwargs:
#       kwargs['name'] = atype
#     name = kwargs['name']

#     # Update groups
#     alist = [*range(self.N, self.N+n)]
#     if name in self.groups:
#       self.groups[name].extend(alist)
#     else:
#       self.groupnames.append(name)
#       self.groups[name] = alist
#     kwargs['gidx'] = self.groupnames.index(name)
    
#     # Noise
#     kwargs['noise'] = noise

#     # --- Initial condition ------------------------------------------------

#     IC = {}
    
#     # --- Positions

#     if type(initial_condition['position']) in [type(None), str]:

#       if initial_condition['position'] in [None, 'random', 'shuffle']:        
#         IC['position'] = [None]*n

#       elif initial_condition['position'] == 'square_lattice':
          
#         pad = initial_condition['padding'] if 'padding' in initial_condition else 0

#         a = ceil(sqrt(n))
#         IC['position'] = [(pad+((k//a)+1/2)/a*(1-2*pad), pad+((k%a)+1/2)/a*(1-2*pad)) for k in range(n)]

#       elif initial_condition['position'] == 'centered':
          
#         IC['position'] = self.engine.boxSize/2 + 0.05*np.random.randn(n, 2)

#     else:

#       IC['position'] = initial_condition['position']

#     # --- Orientations

#     if type(initial_condition['orientation']) in [type(None), str]:

#       if initial_condition['orientation'] in [None, 'random', 'shuffle']:        
#         IC['orientation'] = [None]*n

#     elif type(initial_condition['orientation']) == int or float:      
#       IC['orientation'] = np.full(n, initial_condition['orientation'])
      
#     else:
#       IC['orientation'] = initial_condition['orientation']

#     # --- Speed

#     if type(initial_condition['speed']) == int or float:      
#       IC['speed'] = np.full(n, initial_condition['speed'])
      
#     else:
#       IC['speed'] = initial_condition['speed']
    
#     # --- Add agents -------------------------------------------------------

#     kwargs['engine'] = self.engine

#     for i,k in enumerate(alist):

#       # Agent index
#       kwargs['idx'] = k

#       if damax_range is not None:
#         kwargs['damax'] = damax_range[0] + np.random.rand(1)*(damax_range[1]-damax_range[0])
        
#       # Add the agent based on class name
#       AgentClass = globals()[atype]
#       self.list.append(AgentClass(**kwargs))

#       # Initial conditions
#       self.list[k].setInitialCondition({'position': IC['position'][k-self.N], 
#                                         'orientation': IC['orientation'][k-self.N],
#                                         'speed': initial_condition['speed']})
      
#       # List of group index
#       self.groupIndex.append(kwargs['gidx'])

#       # Density evaluation parameters
#       self.list[k].kde_sigma = self.engine.kde_sigma

#     # Update agent count
#     self.N =  len(self.list)
    
#     if self.engine.verbose is not None:
#       print('→ Added {:d} {:s} agents ({:s}).'.format(n, atype, name))

#   def compile(self):
#     '''
#     Compile all positions and orientations
#     '''

#     return foop(
#       np.array([A.x for A in self.list], dtype=np.float32).flatten(),
#       np.array([A.y for A in self.list], dtype=np.float32).flatten(),
#       np.array([A.a for A in self.list], dtype=np.float32).flatten(),
#       self.engine.boxSize, self.engine.periodic_boundary_condition)

# # === Engine ===============================================================

# class Engine:
#   '''
#   Engine
#   '''

#   # Contructor
#   def __init__(self):

#     # Arena
#     self.boxSize = 1
#     self.periodic_boundary_condition = True

#     # Agents
#     self.agents = Agents(self)

#     # Cues
#     self.cue = None
    
#     # Stop events
#     self.stop_event = None

#     # Display
#     self.window = None
#     self.animation = None

#     # --- Iterations

#     # Number of steps
#     self.steps = None

#     self.verbose = None
#     self.tref = None

#     # --- I/O

#     # Input file
#     self.data_in = None

#     # Trajectories
#     self.store = None
#     self.traj = {}

#     # Records
#     self.records = {}

#     # --- Density estimation

#     # Density estimation lengths
#     self.kde_sigma = 0.1

#     # --- Post-process

#     self.post = {}

#   def input(self, dfile):

#     # Data source
#     self.data_in = dfile

#     # Add agents
#     self.agents.add(self.data_in.Nagents, 'Replay', dataFile = self.data_in)

#   def setup_animation(self, style='dark'):
#     '''
#     Define animation
#     '''

#     self.window = Window('Simple animation', style=style)

#     self.animation = animation.Animation(self)

#     self.window.add(self.animation)

#     # Forbid backward animation
#     self.window.allow_backward = False

#     # Default info agent
#     if self.animation.info_agent is None and len(self.agents.list):
#       self.animation.info_agent = self.agents.list[0]

#   def step(self, iteration):
#     '''
#     One step of the simulation
#     '''

#     if self.verbose is not None and (iteration % self.verbose)==0:
#       print('→ Iteration {:d} ({:.2f} s) ...'.format(iteration, time.time()-self.tref))

#     # Prepare data
#     F = self.agents.compile()

#     # --- Store trajectories
    
#     if self.store is not None:
#       for i in self.store:
#         self.traj[i].append([F.X[i], F.Y[i], F.A[i]])

#     if self.animation is not None:
#       self.animation.update_display(F)

#     # --- Update

#     for i, agent in enumerate(self.agents.list):
#       agent.update(iteration, F, id=i)

#     # --- Post-process -----------------------------------------------------

#     # --- Compensate

#     if 'compensate' in self.post:

#       # Compute average motion
#       zm = np.mean(np.array([a.v*np.exp(1j*a.a[0]) for a in self.agents.list]))

#       match self.post['compensate']:

#         case 'x':

#           # Remove y-component
#           dx = np.real(zm)

#           for a in self.agents.list:
#             a.x -= dx


#         case 'y':

#           # Remove y-component
#           dy = np.imag(zm)

#           for a in self.agents.list:
#             a.y -= dy

#         case 'xy':

#           # Remove both component
#           dx = np.real(zm)
#           dy = np.imag(zm)

#           for a in self.agents.list:
#             a.x -= dx
#             a.y -= dy

#     # --- End of simulation ------------------------------------------------

#     stop = False

#     if self.stop_event is not None:

#       if 'contact_0' in self.stop_event:

#         # Stop on contact with agent 0
#         stop = np.any(np.power(F.X[1:] - F.X[0], 2) + np.power(F.Y[1:] - F.Y[0], 2) < 0.015**2)

#       if 'MISD' in self.stop_event:
          
#         # Minimal Inter-Species Distance (MISD)
#         Ng = len(self.agents.groupnames)
#         MISD = 1
#         for i in range(Ng):
#           for j in range(i+1, Ng):
#             I = self.agents.groups[self.agents.groupnames[i]]
#             J = self.agents.groups[self.agents.groupnames[j]]
#             a = np.stack((F.X[I], F.Y[I]), axis=1)
#             b = np.stack((F.X[J], F.Y[J]), axis=1)
#             D = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)

#             MISD = np.min((MISD, np.min(D)))

#         # Store MISD
#         if 'MISD' in self.records:
#           self.records['MISD'].append(MISD)
#         else:
#           self.records['MISD'] = [MISD]

#         # Stop event
#         if self.stop_event['MISD'](MISD):
#           stop = True

#       if 'mVar' in self.stop_event:

#         Nb = 2000
#         sigma2 = 0.001
#         b = np.linspace(0, 2, Nb)
#         Ng = len(self.agents.groupnames)
#         mV = []

#         for k in range(Ng):

#           I = self.agents.groups[self.agents.groupnames[k]]

#           # --- Minimal variance along x
          
#           # Center
#           dx = np.zeros(Nb)
#           dy = np.zeros(Nb)
#           for i in I:
#             dx += np.exp(-(b-F.X[i])**2/2/sigma2) + np.exp(-(b-F.X[i]-1)**2/2/sigma2)
#             dy += np.exp(-(b-F.Y[i])**2/2/sigma2) + np.exp(-(b-F.Y[i]-1)**2/2/sigma2)
#           cx = (np.argmax(dx)*2/Nb) % 1
#           cy = (np.argmax(dy)*2/Nb) % 1

#           # Variance

#           X = (F.X[I]-cx) % 1
#           Y = (F.Y[I]-cy) % 1
#           X[X>0.5] -= 1
#           Y[Y>0.5] -= 1
#           mV.append(np.sqrt(np.min((np.var(X), np.var(Y)))))
        
#         # Store mVar
#         if 'mVar' in self.records:
#           self.records['mVar'].append(mV)
#         else:
#           self.records['mVar'] = [mV]

#         # Stop event
#         if self.stop_event['mVar'](mV):
#           stop = True

#     if stop or (self.steps is not None and iteration==self.steps-1):

#       # Final number of steps
#       self.steps = iteration+1

#       # End of simulation
#       if self.verbose:
#         print('→ End of simulation @ {:d} steps ({:.2f} s)'.format(self.steps, time.time()-self.tref))

#       # End display
#       if self.animation is not None:
#         self.window.close()

#   def run(self):
#     '''
#     Run the simulation
#     '''

#     # Numpify groups
#     for name in self.agents.groups:
#       self.agents.groups[name] = np.array(self.agents.groups[name])

#     # Reference time
#     self.tref = time.time()

#     # Trajectories
#     if self.store is not None:
#       for i in self.store:
#         self.traj[i] = []

#     # --- Main loop --------------------------------------------------------

#     if self.animation is None:
#       i = 0
#       while self.steps is None or i<self.steps:
#         self.step(i)
#         i += 1

#     else:
#       self.animation.initialize()
#       self.window.show()

    
