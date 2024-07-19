'''
FIELDS
'''

from MIDAS.enums import *

class Fields:
  '''
  Fields class  
  '''

  def __init__(self, engine):

    self.engine = engine
    
    # Number of fields
    self.N = 0

    # Fields
    self.field = []

  def add(self, field):
    '''
    Add a field
    '''

    # Initialization
    field.engine = self.engine
    field.initialization()

    self.field.append(field)
    
    # Update number of fields
    self.N = len(self.field)

    return self.N-1

  def perception(self, **kwargs):
    '''
    Concatenate all field inputs
    '''

    for i, F in enumerate(self.field):
      P = F.perception(**kwargs)
      C = P if i==0 else np.concatenate((C, P))

    return C

  def update(self, **kwargs):
    '''
    Field update
    '''

    for F in self.field:
      F.update(**kwargs)

class Field:

  def __init__(self, shape=None):

    self.engine = None
    self.values = None

    # Field image
    self.shape = None

  def initialization(self, **kwargs):
    '''
    Initialization
    '''

    # --- Default properties

    # Shape
    if isinstance(self.shape, (tuple, list, np.ndarray)):
      pass
    elif self.shape is None:
      self.shape = [100]*self.engine.geom.dimension
    else:
      self.shape = [self.shape]*self.engine.geom.dimension

    # Convert shape to numpy array
    self.shape = np.array(self.shape).astype(int)

    # Default values
    if self.values is None:
      self.values = np.zeros(self.shape)
    
  def pix2x(self, i):
    '''
    Conversion from pixel to x-position
    '''
    
    return (i/self.shape[0] - 1/2)*self.engine.geom.arena_shape[0]
  
  def pix2y(self, j):
    '''
    Conversion from pixel to y-position
    '''
    
    return (j/self.shape[1] - 1/2)*self.engine.geom.arena_shape[1]
  
  def pix2z(self, k):
    '''
    Conversion from pixel to z-position
    '''
    
    return (k/self.shape[2] - 1/2)*self.engine.geom.arena_shape[2]

  def d2pix(self, d):
    '''
    Conversion from x-position to pixel
    '''

    return d/self.engine.geom.arena_shape[0]*self.shape[0]  

  def x2pix(self, x):
    '''
    Conversion from x-position to pixel
    '''

    return (np.round((0.5 + x/self.engine.geom.arena_shape[0])*self.shape[0] - 0.5) % self.shape[0]).astype(int)
  
  def y2pix(self, y):
    '''
    Conversion from y-position to pixel
    '''

    return (np.round((0.5 + y/self.engine.geom.arena_shape[1])*self.shape[1] - 0.5) % self.shape[1]).astype(int)
  
  def z2pix(self, z):
    '''
    Conversion from z-position to pixel
    '''

    return (np.round((0.5 + z/self.engine.geom.arena_shape[2])*self.shape[2] - 0.5) % self.shape[2]).astype(int)

  def perception(self, **kwargs):
    '''
    Field perception
    '''

    # --- Parameters

    nSa = 4

    match self.engine.geom.arena:
      case Arena.RECTANGULAR:
        X = self.engine.geom.arena_shape[0]
        Y = self.engine.geom.arena_shape[1]
        W = self.shape[0]
        H = self.shape[1]
      case Arena.CIRCULAR:
        X = self.engine.geom.arena_shape[0]
        Y = self.engine.geom.arena_shape[0]
        W = self.shape[0]
        H = self.shape[0]

    # --- Pixel indices

    I = np.zeros((self.engine.agents.N, 9), dtype=int)
    J = np.zeros((self.engine.agents.N, 9), dtype=int)
    
    # Reference pixel
    x = (self.engine.agents.pos[:,0]/X + 1/2)*W - 1/2
    y = (self.engine.agents.pos[:,1]/Y + 1/2)*H - 1/2
    I[:,0] = np.round(x) % self.shape[0]
    J[:,0] = np.round(y) % self.shape[1]

    # Other pixels
    I[:,1], J[:,1] = (I[:,0] + 1),  J[:,0]
    I[:,2], J[:,2] = (I[:,0] + 1),  (J[:,0] + 1)
    I[:,3], J[:,3] = I[:,0],        (J[:,0] + 1)
    I[:,4], J[:,4] = (I[:,0] - 1),  (J[:,0] + 1)
    I[:,5], J[:,5] = (I[:,0] - 1),  J[:,0]
    I[:,6], J[:,6] = (I[:,0] - 1),  (J[:,0] - 1)
    I[:,7], J[:,7] = I[:,0],        (J[:,0] - 1)
    I[:,8], J[:,8] = (I[:,0] + 1),  (J[:,0] - 1)
    
    # --- Values

    V = np.tile(np.reshape(self.values[J % H, I % W], [-1, 1, 9]), (1, nSa, 1))

    # --- Coefficients

    # Relative positions (rotated)
    Z_ = (I-x[:,None] +1j*(J-y[:,None]))*np.exp(-1j*self.engine.agents.vel[:,1])[:,None]

    Z = np.tile(np.reshape(Z_, [-1, 1, 9]), (1, nSa, 1))
    A = np.angle(Z) % (2*np.pi)

    # Slice angles
    theta_0_ = np.array([k*2*np.pi/nSa for k in range(nSa)])
    theta_1_ = np.array([(k+1)*2*np.pi/nSa for k in range(nSa)])
    
    theta_0 = np.tile(theta_0_[None,:,None], (self.engine.agents.N, 1, 9))
    theta_1 = np.tile(theta_1_[None,:,None], (self.engine.agents.N, 1, 9))

    # Coefficients
    B = np.logical_and(A>=theta_0, A<theta_1)
    C = np.sum(B*V, axis=2)/np.sum(B, axis=2)

    return C
    