'''
FIELDS
'''

class FieldsBase:
  '''
  Fields base class  
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

    self.field.append(field)

    # Update number of fields
    self.N = len(self.field)

    return self.N-1

  def perception(self, **kwargs):
    '''
    Perception
    '''

    return None

  def update(self, **kwargs):
    '''
    Field update
    '''

    pass