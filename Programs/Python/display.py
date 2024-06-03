import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from functools import partial
import numpy as np
from collections import defaultdict

class visu:
  '''
  Visualization class
  '''

  def __init__(self, E, box=1, arrowSize=0.03):
    
    self.engine = E
    self.s = arrowSize
    self.box = box
    
    self.backend = 'QtAgg'
    self.axesSize = 5
    self.fig = None
    self.ax = None
    self.view = []
    self.Q = []

    # Animation 
    self.anim = None
    self.fps = 25
    self.keep_display_after = False

    self.fMovie = None

    # Default display options
    self.options = {}
    for k,v in self.engine.agents.groups.items():
      self.options[k] = {'color':'w', 'cmap':None, 'size':1}

  def add_view(self, agent=None, reorient=False):
    '''
    Add a view, either global or centered on an agent
    '''

    if agent is None:
      self.view.append({'type':'all'})
    else:
      self.view.append({'type':'one', 'agent':agent, 'reorient':reorient})

  def centerCoordinates(self, F, Idx, view, type):

    out = {'Cm': F.X[Idx]}

    if view['type']=='all':
      out['alpha'] = np.ones(len(Idx))
    else:
      if view['reorient']:
        F = F.center(self.engine.agents.list[view['agent']].x, self.engine.agents.list[view['agent']].y, self.engine.agents.list[view['agent']].a - np.pi/2)
      else:
        F = F.center(self.engine.agents.list[view['agent']].x, self.engine.agents.list[view['agent']].y)
      out['alpha'] = np.isin(Idx, F.near(0.5, include_self=True)).astype(int)

    out['X'] = F.X[Idx]
    out['Y'] = F.Y[Idx]
    out['U'] = self.s*self.options[type]['size']*np.cos(F.A[Idx])
    out['V'] = self.s*self.options[type]['size']*np.sin(F.A[Idx])
    
    return out

  def initialize(self):
    '''
    Initialization of the visual display
    '''

    # --- Figure settings

    mpl.rcParams['toolbar'] = 'None'
    mpl.use(self.backend)
    plt.style.use('dark_background')

    # --- Figure and axes
    self.fig, self.ax = plt.subplots(1, len(self.view), squeeze=False)
    self.fig.tight_layout()

    # Figure on-screen display
    self.fig.set_size_inches(self.axesSize*len(self.view), self.axesSize)
    
    # --- Preparation

    # Misc display variables
    eps = 0.005
    w = 0.5
    h = 1./w*self.s

    # Prepare data
    F = self.engine.agents.compile()

    # --- Display
    
    for i in range(len(self.view)):

      # Axes ratios

      self.ax[0][i].axis('off')
      self.ax[0][i].set_aspect('equal')

      if self.view[i]['type']=='all':

        self.ax[0][i].set_xlim(0-eps,1+eps)
        self.ax[0][i].set_ylim(0-eps,1+eps)

        square = plt.Rectangle([0,0], 1, 1, facecolor='k', edgecolor='w')
        self.ax[0][i].add_patch(square)

      else:

        fv = 0.5 + eps
        self.ax[0][i].set_xlim(-fv,fv)
        self.ax[0][i].set_ylim(-fv,fv)

        circle = plt.Circle([0,0], 0.5, facecolor='k', edgecolor='w', linestyle=':')
        self.ax[0][i].add_patch(circle)
        
      # Quiver plots
      self.Q.append({})

      for name,Idx in self.engine.agents.groups.items():

        Cc = self.centerCoordinates(F, Idx, self.view[i], name)

        if self.options[name]['cmap'] is None:
          self.Q[i][name] = self.ax[0][i].quiver(Cc['X']/self.box, Cc['Y']/self.box, Cc['U'], Cc['V'], 
            pivot='mid', 
            angles='xy', 
            scale_units='xy', 
            scale=1, 
            headwidth=h/2*self.options[name]['size'], 
            headaxislength=h*self.options[name]['size'], 
            headlength=h*self.options[name]['size'], 
            width=w*self.options[name]['size'],
            minlength=0.001,
            color=self.options[name]['color'],
            alpha=Cc['alpha'])

        else:

          self.Q[i][name] = self.ax[0][i].quiver(Cc['X']/self.box, Cc['Y']/self.box, Cc['U'], Cc['V'], Cc['Cm'],
            pivot='mid',
            angles='xy',
            scale_units='xy',
            scale=1,
            headwidth=h/2*self.options[name]['size'],
            headaxislength=h*self.options[name]['size'],
            headlength=h*self.options[name]['size'],
            width=w*self.options[name]['size'],
            minlength=0.001,
            cmap=self.options[name]['cmap'],
            alpha=Cc['alpha'])    

  def update(self, iteration, F):
    '''
    Update visualization
    '''

    self.ax[0][0].set_title('{:06d}'.format(iteration), fontsize='small', loc='left')

    for i in range(len(self.view)):

      for type, Idx in self.engine.agents.groups.items():

        Cc = self.centerCoordinates(F, Idx, self.view[i], type)
        self.Q[i][type].set_offsets(np.array([Cc['X'].flatten()/self.box, Cc['Y'].flatten()/self.box]).T)
        self.Q[i][type].set_UVC(Cc['U'], Cc['V'])
        self.Q[i][type].set_alpha(Cc['alpha'])
      
  def animate(self):
    '''
    Animate the simulation display
    '''

    self.initialize()

    # Check default number of frames in case of export
    if self.fMovie is not None and self.engine.steps is None:
      self.engine.steps = 250

    # Animation
    self.anim = animation.FuncAnimation(self.fig, self.engine.step, 
      init_func = lambda *args: None,
      frames = self.engine.steps, 
      repeat = False,
      interval = 1000/self.fps)

    if self.fMovie is not None:
      self.anim.save(self.fMovie, writer=animation.FFMpegWriter(fps=self.fps))

    plt.show()

  def stop(self):

    # Stop animation
    self.anim.event_source.stop()
      
    # Close figure
    if not self.keep_display_after:
      plt.close(self.fig)