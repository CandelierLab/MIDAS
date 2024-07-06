'''
Storage engine
'''

import os
import numpy as np
import sqlite3

from MIDAS.enums import *
from MIDAS.verbose import cli_Reporter

# === STORAGE ==============================================================

class Storage():

  def __init__(self, db_file, verbose=None):

    self.version = 1
    self.verbose = cli_Reporter() if verbose is None else verbose

    # --- Database source file

    self.db_file = db_file

    # Create directory if not existing
    db_dir = os.path.dirname(self.db_file)
    if not os.path.exists(db_dir): 

      self.verbose(f'Creating folder {db_dir}')
      os.makedirs(db_dir)

    # --- Connection

    if not os.path.exists(self.db_file):
      self.verbose('Creating database')

    self.db_conn = sqlite3.connect(self.db_file)
    self.db_curs = self.db_conn.cursor()

    # --- DB properties
    
    self.db_commit_each_step = False

    # --- Engine properties

    self.dimension = None
    self.Nagents = None

  def initialize(self, engine):
    '''
    Initialize the database based on an Engine
    '''

    # Engine poperties
    self.dimension = engine.geom.dimension
    self.Nagents = engine.agents.N

    # --- File and directory management ------------------------------------

    # Remove if existing
    if os.path.exists(self.db_file):

      self.verbose('Removing existing database')

      os.remove(self.db_file)

      self.verbose('Creating database')

      self.db_conn = sqlite3.connect(self.db_file)
      self.db_curs = self.db_conn.cursor()

    self.verbose('Initializing database')

    # --- Parameters -------------------------------------------------------

    self.db_curs.execute('CREATE TABLE Parameters (key TEXT PRIMARY KEY, value INT NOT NULL);')
    sql_param = 'INSERT INTO Parameters(key,value) VALUES(?,?)'

    match engine.geom.arena:
      case Arena.CIRCULAR:
        periodic = [0, 0, 0]
      case Arena.RECTANGULAR:
        periodic = engine.geom.periodic

    self.db_curs.execute(sql_param, ('db_version', self.version))
    self.db_curs.execute(sql_param, ('dimension', self.dimension))
    self.db_curs.execute(sql_param, ('arena', engine.geom.arena))
    self.db_curs.execute(sql_param, ('arena_X', float(engine.geom.arena_shape[0])))
    self.db_curs.execute(sql_param, ('periodic_X', periodic[0]))

    if self.dimension>1:
      self.db_curs.execute(sql_param, ('arena_Y', float(engine.geom.arena_shape[1])))
      self.db_curs.execute(sql_param, ('periodic_Y', periodic[1]))

    if self.dimension>2:
      self.db_curs.execute(sql_param, ('arena_Z', float(engine.geom.arena_shape[2])))
      self.db_curs.execute(sql_param, ('periodic_Z', periodic[2]))

    if engine.steps is not None:
      self.db_curs.execute(sql_param, ('steps', engine.steps))

    self.db_curs.execute(sql_param, ('Nagents', engine.agents.N))
    self.db_curs.execute(sql_param, ('Ngroups', engine.groups.N))

    # --- Agents -----------------------------------------------------------

    self.db_curs.execute('''CREATE TABLE Agents (
                         id INTEGER PRIMARY KEY, 
                         gid INTEGER NOT NULL
                         );''')

    self.db_curs.executemany('INSERT INTO Agents VALUES (?,?)', 
      np.column_stack((np.arange(engine.agents.N), engine.agents.group)))

    # --- Groups -----------------------------------------------------------

    self.db_curs.execute('''CREATE TABLE Groups (
                         gid INTEGER PRIMARY KEY, 
                         type INTEGER NOT NULL,
                         name TEXT,
                         Nagents INTEGER NOT NULL
                         );''')

    for gid in range(engine.groups.N):
      self.db_curs.execute('INSERT INTO Groups VALUES (?,?,?,?)', 
        (gid,
         engine.groups.atype[gid], 
         engine.groups.names[gid], 
         np.count_nonzero(engine.agents.group==gid)))

    # --- Kinematics -------------------------------------------------------

    self.db_curs.execute(f'''CREATE TABLE Kinematics (
      step INT NOT NULL, 
      id INT NOT NULL, 
      x FLOAT,
      {'y FLOAT,' if self.dimension>1 else ''}
      {'z FLOAT,' if self.dimension>2 else ''}
      v FLOAT,
      {'a FLOAT,' if self.dimension>1 else ''}
      {'b FLOAT,' if self.dimension>2 else ''}
      PRIMARY KEY (step, id));''')

    # --- Store modifications ----------------------------------------------

    self.db_conn.commit()

  def insert_step(self, step, pos, vel):
    '''
    Insert the data of a step in the database
    '''

    sql = 'INSERT INTO Kinematics VALUES (?,?'+ ',?,?'*self.dimension + ')'

    self.db_curs.executemany(sql, np.hstack(
      (np.full((self.Nagents,1), step),
      np.arange(self.Nagents)[:,None], 
      pos, 
      vel) ))

    if self.db_commit_each_step:
      self.db_conn.commit()

