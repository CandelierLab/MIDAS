import os
import time
import sqlite3
import numpy as np

os.system('clear')

# === Parameters ===========================================================

dbfile = '/home/raphael/Science/Projects/CM/MovingAgents/Data/RIPO/test.db'

# ==========================================================================

# Remove if existing
if os.path.exists(dbfile): os.remove(dbfile)

conn = sqlite3.connect(dbfile)
C = conn.cursor()

# --- Parameters -----------------------------------------------------------

C.execute("""CREATE TABLE Parameters (
  key TEXT PRIMARY KEY, 
  value INT NOT NULL);
  """)

C.execute('INSERT INTO Parameters(key,value) VALUES(?,?)', ('dimension', 2))
C.execute('INSERT INTO Parameters(key,value) VALUES(?,?)', ('arena', 0))
C.execute('INSERT INTO Parameters(key,value) VALUES(?,?)', ('periodic_X', 1))
C.execute('INSERT INTO Parameters(key,value) VALUES(?,?)', ('steps', 100))
C.execute('INSERT INTO Parameters(key,value) VALUES(?,?)', ('Nagents', 100))

# --- Kinematics -----------------------------------------------------------

C.execute("""CREATE TABLE Kinematics (
  t INT NOT NULL, 
  id INT NOT NULL, 
  x FLOAT,
  y FLOAT,
  r FLOAT,
  a FLOAT,          
  PRIMARY KEY (t, id));
  """)

test = np.random.rand(1000,6)

tref = time.perf_counter()

C.executemany('INSERT INTO Kinematics VALUES (?,?,?,?,?,?)', test)

print(time.perf_counter()-tref)

conn.commit()