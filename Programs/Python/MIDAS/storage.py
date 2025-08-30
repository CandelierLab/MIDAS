'''
Storage
'''

import os
import numpy as np
import sqlite3
from rich import print

import MIDAS

class storage():

  # ════════════════════════════════════════════════════════════════════════
  #                               INITIALIZATION
  # ════════════════════════════════════════════════════════════════════════
    
  # ────────────────────────────────────────────────────────────────────────
  def __init__(self, db_file, verbose=False):

    # ─── Definitions

    self.version = '1.1.0'
    self.verbose = verbose

    self.engine = None
    self.dimension = None
    self.N = None

    # ─── Database source file

    self.db_file = db_file

    # Create directory if not existing
    db_dir = os.path.dirname(self.db_file)
    if not os.path.exists(db_dir):

      if self.verbose: print(f'Creating folder {db_dir}')
      os.makedirs(db_dir)

    # Create database if not existing
    if not os.path.exists(self.db_file) and self.verbose:
      print('Creating database')

    # ─── Connection

    self.db_conn = sqlite3.connect(self.db_file)
    self.db_curs = self.db_conn.cursor()

    # ─── DB properties
    
    self.commit_frequency = MIDAS.COMMIT.AT_THE_END

    # ─── Engine properties

    self.dimension = None
    self.Nagents = None

  # ────────────────────────────────────────────────────────────────────────
  def initialize(self):
    '''
    Initialize the database based on an engine

    NB: Do not initialize when reading the database.
    '''

    # ─── Remove database if existing ───────────

    if os.path.exists(self.db_file):

      if self.verbose: print('Removing existing database')
      os.remove(self.db_file)

      # Re-connect
      self.db_conn = sqlite3.connect(self.db_file)
      self.db_curs = self.db_conn.cursor()

    # Engine poperties
    self.dimension = int(self.engine.geometry.dimension)
    self.N = int(self.engine.agents.N)

    # ─── Parameters ────────────────────────────

    self.db_curs.execute('CREATE TABLE Parameters (key TEXT PRIMARY KEY, value REAL NOT NULL);')
    sql_param = 'INSERT INTO Parameters(key,value) VALUES(?,?)'

    self.db_curs.execute(sql_param, ('db_version', self.version))
    self.db_curs.execute(sql_param, ('dimension', self.dimension))

    # Arena
    arena = self.engine.geometry.arena
    self.db_curs.execute(sql_param, ('arena_type', arena.type))

    match arena.type:

      case MIDAS.ARENA.CIRCULAR:

        self.db_curs.execute(sql_param, ('arena_name', 'CIRCULAR'))
        periodic = [False]*3

      case MIDAS.ARENA.RECTANGULAR:

        self.db_curs.execute(sql_param, ('arena_name', 'RECTANGULAR'))
        periodic = arena.periodic

    self.db_curs.execute(sql_param, ('arena_X', float(arena.shape[0])))
    self.db_curs.execute(sql_param, ('periodic_X', periodic[0]))

    if self.dimension>1:
      self.db_curs.execute(sql_param, ('arena_Y', float(arena.shape[1])))
      self.db_curs.execute(sql_param, ('periodic_Y', periodic[1]))

    if self.dimension>2:
      self.db_curs.execute(sql_param, ('arena_Z', float(arena.shape[2])))
      self.db_curs.execute(sql_param, ('periodic_Z', periodic[2]))

    if self.engine.steps is not None:
      self.db_curs.execute(sql_param, ('steps', self.engine.steps))

    self.db_curs.execute(sql_param, ('N', self.N))

    # ─── Agents ────────────────────────────────

    self.db_curs.execute('''CREATE TABLE Agents (
                         id INTEGER PRIMARY KEY, 
                         gid INTEGER NOT NULL
                         );''')

    self.db_curs.executemany('INSERT INTO Agents VALUES (?,?)', 
      np.column_stack((np.arange(self.N), self.engine.agents.group)).tolist())

    # ─── Groups ────────────────────────────────

    self.db_curs.execute('''CREATE TABLE Groups (
                         gid INTEGER PRIMARY KEY, 
                         class TEXT,
                         name TEXT,
                         N INTEGER NOT NULL
                         );''')

    for group in self.engine.group:

      self.db_curs.execute('INSERT INTO Groups VALUES (?,?,?,?)', 
        (group.id, group.__class__.__name__, group.name, group.N))

    # ─── Kinematics ────────────────────────────

    self.db_curs.execute(f'''CREATE TABLE Dynamics (
      step INT NOT NULL, 
      id INT NOT NULL, 
      x FLOAT,
      {'y FLOAT,' if self.dimension>1 else ''}
      {'z FLOAT,' if self.dimension>2 else ''}
      v FLOAT,
      {'a FLOAT,' if self.dimension>1 else ''}
      {'b FLOAT,' if self.dimension>2 else ''}
      PRIMARY KEY (step, id));''')

    # ─── Store modifications ───────────────────

    self.db_conn.commit()

  # ════════════════════════════════════════════════════════════════════════
  #                                WRITING
  # ════════════════════════════════════════════════════════════════════════

  # ────────────────────────────────────────────────────────────────────────
  def insert_step(self, step, pos, vel):
    '''
    Insert the data of a step in the database
    '''

    sql = 'INSERT INTO Dynamics VALUES (?,?'+ ',?,?'*self.dimension + ')'

    self.db_curs.executemany(sql, np.hstack(
      (np.full((self.N,1), step),
      np.arange(self.N)[:,None], 
      pos, 
      vel) ))

    if self.commit_frequency == MIDAS.COMMIT.EVERY_1_STEP:
      self.db_conn.commit()

  # ════════════════════════════════════════════════════════════════════════
  #                                READING
  # ════════════════════════════════════════════════════════════════════════

  # ────────────────────────────────────────────────────────────────────────
  def get_time(self, step):
    
    res = self.db_curs.execute(f'SELECT * FROM Dynamics WHERE step={step}').fetchall()
    return np.array(res)

  # ────────────────────────────────────────────────────────────────────────
  def get_trajectory(self, id):
    
    res = self.db_curs.execute(f'SELECT * FROM Dynamics WHERE id={id}').fetchall()
    return np.array(res)
