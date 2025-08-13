'''
MIDAS animation
'''

import numpy as np
from scipy.ndimage import gaussian_filter
import anim

import MIDAS

class animation(anim.plane.canva):

  # ════════════════════════════════════════════════════════════════════════
  #                              INITIALIZATION
  # ════════════════════════════════════════════════════════════════════════

  # ────────────────────────────────────────────────────────────────────────
  def __init__(self, 
               engine = None,
               agents = MIDAS.ANIMATION_AGENTS.NONE,
               field = MIDAS.ANIMATION_FIELD.NONE,
               title = 'MIDAS',
               style = 'dark'):
    '''
    Constructor
    '''

    # Definitions
    self.engine = engine
    self.agents = agents
    self.field = field

    # Window
    self.window = anim.window(title, style=style)

  # ────────────────────────────────────────────────────────────────────────
  def initialize(self):
    
    # ─── Setup

    self.dimension = self.engine.geometry.dimension

    # Animation constructor
    super().__init__(self.window,
                     boundaries=[np.array([-1, 1])*self.engine.geometry.arena.shape[0]/2,
                                 np.array([-1, 1])*self.engine.geometry.arena.shape[1]/2],
                     display_boundaries=False)

    # Add animation to window
    self.window.add(self)

    # ─── Boundaries
    ''' We need custom boundaries for periodic boundary conditions and circular arenas. '''

    self.set_boundaries()



    # # --- Agents and field -------------------------------------------------

    # self.set_agents(agents)
    # self.field = field

    # # --- Agent options

    # # Trajectory traces
    # self.trace_duration = None

    # self.group_options = {}

    # # Default group options
    # for k in self.engine.groups.names:
    #   self.group_options[k] = {
    #     'color': ('white', 'white'), 
    #     'cmap': None,
    #     'cmap_on': 'x0',
    #     'cmap_dynamic': None,
    #     'size': 0.007
    #   }

    # # --- Field options

    # self.field_options = {}

    # self.field_options['resolution'] = np.array([500, 500])
    # self.field_options['sigma'] = 5
    # self.field_options['range'] = [0, 1]
    # self.field_options['cmap'] = 'turbo'

    # # --- Misc properties --------------------------------------------------

    # self.W = self.engine.geom.arena_shape[0]
    # self.H = self.engine.geom.arena_shape[1]
    # self.is_running = True

    # # Shift and grid
    # self.shift = np.zeros((2))

    # self.gridsize = None
    # self.ngrid = None


    # === Agents ===========================================================

    # if self.l_agents is not None:

    #   # Agent's triangle shape
    #   pts = np.array([[1,0],[-0.5,0.5],[-0.5,-0.5]])

    #   for i in self.l_agents:
        
    #     # Group options
    #     opt = self.group_options[self.engine.groups.names[int(self.engine.agents.group[i])]]

    #     # --- Color

    #     # Colormap and color
    #     if opt['cmap'] is None:
    #       color = opt['color']
    #     else:
    #       color = None
    #       cmap = Colormap(name=opt['cmap'])

    #     if color is None:
    #       match opt['cmap_on']:

    #         case 'index': # Color on index
    #           n = np.count_nonzero(self.engine.agents.group==self.engine.agents.group[i])
    #           cmap.range = [0, n-1]
    #           clrs = [cmap.qcolor(i)]*2

    #         case 'x0': # Color on x-position (default)
    #           cmap.range = self.boundaries['x']
    #           clrs = [cmap.qcolor(self.engine.agents.pos[i,0])]*2

    #         case 'y0': # Color on y-position   
    #           cmap.range = self.boundaries['y']         
    #           clrs = [cmap.qcolor(self.engine.agents.pos[i,1])]*2

    #         case 'z0': # Color on z-position            
    #           cmap.range = self.boundaries['z']
    #           clrs = [cmap.qcolor(self.engine.agents.pos[i,2])]*2

    #     elif isinstance(color, (tuple, list)):
    #       clrs = color

    #     else:
    #       clrs = (color, None)

    #     # --- Shape

    #     if self.engine.groups.atype[int(self.engine.agents.group[i])]==Agent.FIXED:
    #       '''
    #       Fixed agents
    #       '''

    #       self.add(circle, i,
    #         position = self.engine.agents.pos[i,:],
    #         radius = 0.0035,
    #         colors = clrs,
    #         zvalue = 2
    #       )

    #     else:
    #       '''
    #       Moving agents
    #       '''

    #       self.add(polygon, i,
    #         position =  self.engine.agents.pos[i,:],
    #         orientation = self.engine.agents.vel[i,1],
    #         points = pts*opt['size'],
    #         colors = clrs,
    #         zvalue = 10
    #       )

    #     # --- Traces
        
    #     if self.trace_duration is not None:

    #       trace = [self.engine.agents.pos[i,:]]*self.trace_duration

    #       # Semi-transparent color
    #       clr = QColor(clrs[0])
    #       clr.setAlpha(100)

    #       # Trace paths
    #       self.add(path, f'{i:d}_trace',
    #         position = -self.engine.geom.arena_shape/2,
    #         orientation = 0,
    #         points = trace,
    #         colors = (None, clr),
    #         thickness = 3,
    #         zvalue = 9
    #       )

    # # === Field ============================================================

    # if self.field is not AnimField.NONE:

    #   # Colormap
    #   field_colormap = Colormap(self.field_options['cmap'], range=self.field_options['range'], ncolors=256)

    #   # Image container
    #   self.add(field, 'field',
    #            position = -self.engine.geom.arena_shape/2,
    #            cmap = field_colormap,
    #            flip_vertical = True,
    #            zvalue = 0,
    #            )
      
    #   # Colorbar
    #   self.window.information.add(colorbar, 'field_colorbar',
    #                               colormap = field_colormap,
    #                               insight = True,
    #                               height = 0.5,
    #                               nticks = 2)
       
    # # === Misc display items ===============================================

    # # Grid

    # if self.gridsize is not None:

    #   self.ngrid = np.floor(np.array([self.W, self.H])/self.gridsize).astype(int)

    #   # Vertical lines
    #   for i in range(self.ngrid[0]):
    #     self.add(line, f'grid_v{i}',
    #             points = [[0, 0], [0, 0]],
    #             color = 'gray',
    #             linestyle = ':',
    #             thichness = 1,
    #             zvalue = 1)
        
    #   # Horizontal lines
    #   for i in range(self.ngrid[1]):
    #     self.add(line, f'grid_h{i}',
    #             points = [[0, 0], [0, 0]],
    #             color = 'gray',
    #             linestyle = ':',
    #             thichness = 1,
    #             zvalue = 1)

    # # Update display
    # self.update_display()

  def set_agents(self, agents):
    '''
    Set the list of agents to display
    '''

    match agents:

      case AnimAgents.NONE:
        self.l_agents = []

      case AnimAgents.SUBSET_10:
        self.l_agents = np.unique(np.round(np.linspace(0, self.engine.agents.N-1, 10)).astype(int)) if self.engine.agents.N>10 else np.array(list(range(self.engine.agents.N)), dtype=int)

      case AnimAgents.SUBSET_100:
        self.l_agents = np.unique(np.round(np.linspace(0, self.engine.agents.N-1, 100)).astype(int)) if self.engine.agents.N>100 else np.array(list(range(self.engine.agents.N)), dtype=int)

      case AnimAgents.ALL:
        self.l_agents = np.array(list(range(self.engine.agents.N)), dtype=int)

      case _:
        self.l_agents = np.array(agents, dtype=int)

  # ────────────────────────────────────────────────────────────────────────
  def set_boundaries(self):
    '''
    Set the boundaries
    '''

    # Arena
    arena = self.engine.geometry.arena

    match arena.type:

      case MIDAS.ARENA.CIRCULAR:

        self.item.boundary = anim.plane.circle(
          position = [0, 0],
          radius = arena.shape[0]/2,
          stroke = 'grey',
          color = None,
          zvalue = 1
        )

      case MIDAS.ARENA.RECTANGULAR:

        bounds_x = np.array([-1, 1])*arena.shape[0]/2
        bounds_y = np.array([-1, 1])*arena.shape[1]/2

        # ─── X-axis

        # Periodicity
        lstyle = '--' if arena.periodic[0] else '-'
           
        self.item.boundary_left = anim.plane.line(
          position = [bounds_x[0], bounds_y[0]],
          dimension = [0,  arena.shape[1]],
          color = 'grey',
          linestyle = lstyle,
          thickness = 0,
          zvalue = 1)
          
        self.item.boundary_right = anim.plane.line(
          position = [bounds_x[1], bounds_y[0]],
          dimension = [0,  arena.shape[1]],
          color = 'grey',
          linestyle = lstyle,
          thickness = 0,
          zvalue = 1)

        # ─── Y-axis

        # Periodicity
        lstyle = '--' if arena.periodic[1] else '-'
           
        self.item.boundary_bottom = anim.plane.line(
          position = [bounds_x[0], bounds_y[0]],
          dimension = [arena.shape[0], 0],
          color = 'grey',
          linestyle = lstyle,
          thickness = 0,
          zvalue = 1)
          
        self.item.boundary_top = anim.plane.line(
          position = [bounds_x[0], bounds_y[1]],
          dimension = [arena.shape[0], 0],
          color = 'grey',
          linestyle = lstyle,
          thickness = 0,
          zvalue = 1)

  # ------------------------------------------------------------------------
  #   Informations
  # ------------------------------------------------------------------------

  # def add_info(self, agent=None):

  #   # Define agent
  #   if agent is None:
  #     agent = self.info_agent

  #   # Get information
  #   info = agent.information()

  #   self.engine.window.information.add(text, 'info',
  #     stack = True,
  #     string = info,
  #     color = 'white',
  #     fontsize = 10,
  #   )

    # match self.engine.animation.options[self.name]['dynamic_cmap']:
  
    #   case 'speed':
    #     self.engine.animation.colormap.range = [self.v_min, self.v_max]
    #     self.engine.animation.add_insight_colorbar()

    #   case 'density':
    #     self.engine.animation.colormap.range = [1, 20]
    #     self.engine.animation.add_insight_colorbar()

  # # # def add_info_weights(self, agent=None):
  # # #   '''
  # # #   Information over weights displayed in a piechart style
  # # #   '''
    
  # # #   if agent is None:
  # # #     agent = self.info_agent

  # # #   agent.add_info_weights()
  # # #   agent.update_info_weights()

  # ------------------------------------------------------------------------
  #   Updates
  # ------------------------------------------------------------------------

  def update(self, t):
    '''
    Update method
    '''

    # Update timer display
    super().update(t)

    # Compute step
    self.engine.step(t.step)

    # Update shift
    self.update_shift()

    #  Update display
    self.update_display(t=t)

  def update_shift(self, **kwargs):
    '''
    Update the display shift
    '''

    pass

  def update_display(self, **kwargs):
    '''
    Update display
    '''
    
    # === Agents ===========================================================

    if self.l_agents is not None:

      # Positions
      x = self.engine.agents.pos[:,0] + self.shift[0]
      y = self.engine.agents.pos[:,1] + self.shift[1]

      # Periodicity
      if self.engine.geom.arena==Arena.RECTANGULAR:
        if self.engine.geom.periodic[0]: x = ((x + self.W/2) % self.W) - self.W/2
        if self.engine.geom.periodic[1]: y = ((y + self.H/2) % self.H) - self.H/2

      for i in self.l_agents:

        # Skip fixed agents
        if self.engine.groups.atype[int(self.engine.agents.group[i])]==Agent.FIXED: continue

        # Position
        self.item[i].position = [x[i], y[i]]

        # Orientation
        self.item[i].orientation = self.engine.agents.vel[i,1]

        # --- Traces

        if self.trace_duration is not None:

          # Previous trace
          trace = np.array(self.item[f'{i:d}_trace'].points)

          # Roll new trace
          trace = np.roll(trace, 1, axis=0)
          trace[0,0] = x[i]
          trace[0,1] = y[i]

          # Periodic boundary conditions
          if self.engine.geom.arena==Arena.RECTANGULAR:
          
            if self.engine.geom.periodic[0]:
              trace[:,0] = np.unwrap(trace[:,0], period=self.engine.geom.arena_shape[0], axis=0)
              I = np.logical_or(trace[:,0]<-self.engine.geom.arena_shape[0]/2, trace[:,0]>self.engine.geom.arena_shape[0]/2)
              trace[I,0] = np.nan
              trace[I,1] = np.nan

            if self.engine.geom.periodic[1]:
              trace[:,1] = np.unwrap(trace[:,1], period=self.engine.geom.arena_shape[1], axis=0)
              I = np.logical_or(trace[:,1]<-self.engine.geom.arena_shape[1]/2, trace[:,1]>self.engine.geom.arena_shape[1]/2)
              trace[I,0] = np.nan
              trace[I,1] = np.nan
            
          # Update trace
          self.item[f'{i:d}_trace'].points = trace

    # === Field ============================================================

    match self.field:

      case AnimField.NONE:
        pass

      case AnimField.DENSITY:
        '''
        The 2D integral of the density should be N, the number of agents.
        '''

        # Raw density
        Img = np.zeros((self.field_options['resolution'][1], self.field_options['resolution'][0]))
        
        for k in range(self.engine.agents.N):

          x = (self.engine.agents.pos[k][0] + self.shift[0])/self.engine.geom.arena_shape[0] + 0.5
          y = (self.engine.agents.pos[k][1] + self.shift[1])/self.engine.geom.arena_shape[1] + 0.5

          # Periodicity
          if self.engine.geom.arena==Arena.RECTANGULAR:
            if self.engine.geom.periodic[0]: x = x % 1
            if self.engine.geom.periodic[1]: y = y % 1

          i = round(x*self.field_options['resolution'][0] - 0.5) % self.field_options['resolution'][0]
          j = round(y*self.field_options['resolution'][1] - 0.5) % self.field_options['resolution'][1]

          Img[j,i] += 1
          
        # Gaussian smooth
        Res = gaussian_filter(Img, (self.field_options['sigma'], self.field_options['sigma']))

        # Update displayed field
        self.item['field'].field = Res

      case AnimField.POLARITY:

        # !! TODO !!
        pass

      case _:

        if self.engine.fields is not None and not isinstance(self.field, str) and self.field<self.engine.fields.N:
          self.item['field'].image = self.engine.fields.field[self.field].values

    # === Misc =============================================================

    # Grid

    if self.ngrid is not None:

      # Horizontal lines
      for i in range(self.ngrid[1]):
        yg = (i*self.gridsize + self.shift[1]) % self.H - self.H/2
        self.item[f'grid_h{i}'].points = [[-self.W/2, yg], [self.W/2, yg]]

      # Vertical lines
      for i in range(self.ngrid[0]):
        xg = (i*self.gridsize + self.shift[0]) % self.W - self.W/2
        self.item[f'grid_v{i}'].points = [[xg, -self.H/2], [xg, self.H/2]]

  def stop(self):
    '''
    Method triggered on animation exit
    '''
    pass
    # if self.is_running:
    #   self.engine.end()