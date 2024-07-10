from Animation.Animation_2d import *
from Animation.Colormap import *
from MIDAS.enums import *

class InformationBase:

  def __init__(self, engine):

    # Definitions
    self.engine = engine
    self.qanim = self.engine.window.information

    self.display()
    
  # ========================================================================
  def display(self):
    '''
    Display information. 
    To overload for customization
    '''

    self.display_standard()

  # ========================================================================
  def display_standard(self):
    '''
    Display standard MIDAS information
    '''

    # Compact mode
    self.qanim.stack['vpadding'] = -0.005

    # --- Arena

    s = 'Arena '
    match self.engine.geom.arena:
      case Arena.RECTANGULAR:
        s += f'{self.engine.geom.arena_shape}'
      case Arena.CIRCULAR:
        s += f'D={self.engine.geom.arena_shape[0]}'

    self.qanim.add(text, 'arena', stack = True, string = s)

    # --- Agents

    s = f'N= {self.engine.agents.N}'
    match self.engine.geom.arena:
      case Arena.RECTANGULAR:
        s += f' ({self.engine.agents.N/np.prod(self.engine.geom.arena_shape)})'
      case Arena.CIRCULAR:
        s += f' ({self.engine.agents.N/np.pi/(self.engine.geom.arena_shape[0]/2)**2})'

    self.qanim.add(text, 'arena', stack = True, string = s)

    # --- Groups

    # for gid in range(self.engine.groups.N):

    #   # Group name
    #   s = f'Group [{self.engine.groups.names[gid]}]'

    #   # Append group infos
    #   self.qanim.add(text, f'group_{gid}', stack = True, string = s)

# ##########################################################################
#                             COMPOSITES
# ##########################################################################

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
