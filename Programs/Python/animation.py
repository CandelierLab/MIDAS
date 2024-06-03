import re
from agents import *

from AE.Display.Animation.Animation_2d import *
from AE.Display.Colormap import *

class piechart(composite):
  """
  Piechart composite element
  """

  def __init__(self, animation, name, **kwargs):
    """
    Piechart constructor
    """  

    super().__init__(animation, name, **kwargs)

    # --- Arguments

    self.radius = kwargs['radius'] if 'radius' in kwargs else 0.1
    if 'values' in kwargs:
      self.values = kwargs['values']
      self.ns = len(self.values[0])
    else:
      self.values = None
      self.ns = kwargs['ns'] if 'ns' in kwargs else 1
    self.cmap = kwargs['cmap'] if 'cmap' in kwargs else Colormap()
    self.delta = kwargs['delta'] if 'delta' in kwargs else 0
    self.fontsize = kwargs['fontsize'] if 'fontsize' in kwargs else 8
    self.ratio = kwargs['ratio'] if 'ratio' in kwargs else 0.6

    # --- Items

    # Orientation arrow
    aname = self.name + '_dir'

    self.animation.add(polygon, aname, parent=self.name,
      points = [[self.radius, 0.01],[self.radius/2,-self.radius/2],[self.radius*3/2,-self.radius/2]],
      colors = ('grey', None),
    )

    self.animation.item[aname].transformPoint = (self.radius, -self.radius)
    self.animation.item[aname].orientation = self.delta

    for i in range(self.ns):

      # --- Background pie segment

      # Color
      color = self.val2color(self.values[1][i])

      self.animation.add(circle, self.name + '_pie_{:d}_background'.format(i), parent=self.name,
        position = (self.radius, -self.radius),
        radius = self.radius,
        span = (np.pi/2*(1+i*4/self.ns), 2*np.pi/self.ns),
        colors = (color, None)
      )

      # --- Values

      if self.values is not None:

        name = self.name + '_text_{:d}'.format(i)
        theta = (4*i+2+self.ns)*np.pi/2/self.ns
        x = self.radius*(np.cos(theta) + 1)
        y = self.radius*(np.sin(theta) - 1)

        self.animation.add(text, name, parent=self.name,
          position = (x, y),
          string = '{:.01f}'.format(self.values[1][i]),
          fontsize = self.fontsize,
          color = 'black',
          center = (False, True)
        )

        if theta>3*np.pi/2:
          self.animation.item[name].position = (x-self.animation.item[name].width(), y)
          self.animation.item[name].transformPoint = (self.animation.item[name].width(), -self.animation.item[name].height()/2)
          self.animation.item[name].orientation = float(theta)
        else:
          self.animation.item[name].transformPoint = (0,-self.animation.item[name].height()/2)
          self.animation.item[name].orientation = float(theta-np.pi)

      # --- Foreground pie segment

      # Color
      color = self.val2color(self.values[0][i])

      self.animation.add(circle, self.name + '_pie_{:d}_foreground'.format(i), parent=self.name,
        position = (self.radius, -self.radius),
        radius = self.radius*self.ratio,
        span = (np.pi/2*(1+i*4/self.ns), 2*np.pi/self.ns),
        colors = (color, None)
      )

      # --- Values

      if self.values is not None:

        name = self.name + '_text_{:d}'.format(i)
        theta = (4*i+2+self.ns)*np.pi/2/self.ns
        x = self.radius*(np.cos(theta)*self.ratio + 1)
        y = self.radius*(np.sin(theta)*self.ratio - 1)

        self.animation.add(text, name, parent=self.name,
          position = (x, y),
          string = '{:.01f}'.format(self.values[0][i]),
          fontsize = self.fontsize,
          color = 'black',
          center = (False, True)
        )

        if theta>3*np.pi/2:
          self.animation.item[name].position = (x-self.animation.item[name].width(), y)
          self.animation.item[name].transformPoint = (self.animation.item[name].width(), -self.animation.item[name].height()/2)
          self.animation.item[name].orientation = float(theta)
        else:
          self.animation.item[name].transformPoint = (0,-self.animation.item[name].height()/2)
          self.animation.item[name].orientation = float(theta-np.pi)

  def val2color(self, vcol):

    if vcol>self.cmap.range[1]: vcol = self.cmap.range[1]
    if vcol<self.cmap.range[0]: vcol = self.cmap.range[0]
    vcol = np.sign(vcol)*np.sqrt(np.abs(vcol))

    return self.cmap.qcolor(vcol)

class Animation(Animation_2d):
    
  def __init__(self, engine):

    # Define engine
    self.engine = engine

    # Animation constructor
    super().__init__()

    # Default display options
    self.options = {}
    for k,v in self.engine.agents.groups.items():
      self.options[k] = {
        'color': 'white', 
        'cmap': None,
        'cmap_distribution': 'x',
        'dynamic_cmap': None,
        'mapColorOn': 'x0',
        'size': 0.015
      }

    # Trajectory trace
    self.trace_duration = None

    # Default info agent
    self.info_agent = None

    # Colormaps
    self.pie_cmap = Colormap('coolwarm')
    self.pie_cmap.range = [-1.5,1.5]
    self.colormap = Colormap('cool')

    # Timeline
    self.timeline = {}
    
  def initialize(self):

    # === Agents ===========================================================

    # Define padding
    padding = np.max([self.options[k]['size'] for k in self.options])    
    self.setPadding(padding)

    # Agent shape
    pts = np.array([[1,0],[-0.5,0.5],[-0.5,-0.5]])

    for name, Idx in self.engine.agents.groups.items():

      # Color and colormaps
      if self.options[name]['cmap'] is None:
        color = self.options[name]['color']
      else:
        color = None
        cmap = Colormap(name=self.options[name]['cmap'])

      for i in Idx:

        # Agent
        Ag = self.engine.agents.list[i]

        if color is None:

          match self.options[name]['cmap_distribution']:
            case 'index':
              clrs = (cmap.qcolor(i/len(Idx)), None)
            case _:
              # Default: x-position
              clrs = (cmap.qcolor(Ag.x), None)

        elif isinstance(color, tuple):
          clrs = color

        else:
          clrs = (color, None)

        if type(Ag)==Fixed:
          self.add(circle, i,
            position = [Ag.x, Ag.y],
            radius = 0.005,
            colors = clrs,
            zvalue=-1
          )

        # elif type(Ag)==RPA_mpct and (Ag.rHb[Ag.gidx]>0 or Ag.rHf[Ag.gidx]>0):

        #   r = np.max((Ag.rHb[Ag.gidx], Ag.rHf[Ag.gidx]))

        #   self.add(group, i,
        #     position = [Ag.x, Ag.y],
        #     orientation = Ag.a
        #   )

        #   self.add(circle, 'circle_{:d}'.format(i), parent=i,
        #     position = [0, 0],
        #     radius = r/2,
        #     colors = clrs,
        #     thickness = 0
        #   )

        #   self.add(polygon, 'arrow_{:d}'.format(i), parent=i,
        #     position = [0, 0],
        #     points = pts*self.options[name]['size']*r*35,
        #     colors = ('Back','None'),
        #   )

        else:

          self.add(polygon, i,
            position = [Ag.x, Ag.y],
            orientation = Ag.a,
            points = pts*self.options[name]['size'],
            colors = clrs,
          )

        # === Traces =======================================================
          
        if self.trace_duration is not None:

          # Initialize trace coordinates
          Ag.trace = np.ones((self.trace_duration,1))*np.array([Ag.x, Ag.y])
      
          # Trace polygon
          self.add(path, f'{i:d}_trace',
            position = [0, 0],
            orientation = 0,
            points = Ag.trace,
            colors = (None, clrs[0]),
            thickness = 3
          )

  def time_str(self):

    s = '<p>step {:06d}</p>'.format(self.step)

    # Grey zeros
    s = re.sub(r'( )([0]+)', r'\1<span style="color:grey;">\2</span>', s)

    # Agents that are not fixed
    s += '<p>{:d} agents</p>'.format(np.count_nonzero([type(a)!=Fixed for a in self.engine.agents.list]))

    if not self.engine.periodic_boundary_condition:
      s += '<p style="font-size:20px;">Bouncing boundary condition</p>'

    return s

  def add_info(self, agent=None):

    # Define agent
    if agent is None:
      agent = self.info_agent

    # Get information
    info = agent.information()

    self.engine.window.information.add(text, 'info',
      stack = True,
      string = info,
      color = 'white',
      fontsize = 10,
    )

    # match self.engine.animation.options[self.name]['dynamic_cmap']:
  
    #   case 'speed':
    #     self.engine.animation.colormap.range = [self.v_min, self.v_max]
    #     self.engine.animation.add_insight_colorbar()

    #   case 'density':
    #     self.engine.animation.colormap.range = [1, 20]
    #     self.engine.animation.add_insight_colorbar()

  def add_info_weights(self, agent=None):
    '''
    Information over weights displayed in a piechart style
    '''
    
    if agent is None:
      agent = self.info_agent

    agent.add_info_weights()
    agent.update_info_weights()

  def add_info_colorbar(self):

    pass
    # self.add(colorbar, 'Cb',
    #   insight = True,
    #   height = 'fill',
    #   nticks = 2
    # )

  def update(self, t):
    
    # Update timer display
    super().update(t)

    # Timeline changes
    if t.step in self.timeline:
      exec(self.timeline[t.step])
      self.info_agent.update_info_weights()

    # Compute step
    self.engine.step(t.step)

    # Update traces
    if self.trace_duration is not None:
      for i, Ag in enumerate(self.engine.agents.list):
        Ag.trace = np.roll(Ag.trace, 1, axis=0)
        Ag.trace[0,0] = Ag.x
        Ag.trace[0,1] = Ag.y

        # Periodic boundary conditions
        Ag.trace = np.unwrap(Ag.trace, period=1, axis=0)
        Ag.trace[Ag.trace<0] = 0
        Ag.trace[Ag.trace>1] = 1
        

  def update_display(self, F):
    '''
    Update display
    '''
    
    mx = 0
    Mx = 0

    for i, Ag in enumerate(self.engine.agents.list):

      # Position
      self.item[i].position = [F.X[i], F.Y[i]]

      # Orientation
      self.item[i].orientation = F.A[i]

      # Color
      match self.options[self.engine.agents.list[i].name]['dynamic_cmap']:
        case 'speed':
          self.item[i].colors = (self.colormap.qcolor(self.engine.agents.list[i].v), None)
        case 'density':
          if self.engine.agents.list[i].density is not None:
            self.item[i].colors = (self.colormap.qcolor(self.engine.agents.list[i].density), None)
        case 'custom':
          self.item[i].colors = (self.colormap.qcolor(self.engine.agents.list[i].get_color()), None)
    
      # Traces
      if self.trace_duration is not None:
        self.item[f'{i:d}_trace'].points = Ag.trace
            
    #   mx = min(mx, np.min(Ag.trace[:,0]))
    #   Mx = max(Mx, np.max(Ag.trace[:,0]))

    # print(mx, Mx)