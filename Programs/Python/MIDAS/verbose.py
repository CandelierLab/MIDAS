'''
Verbose: display informations in the command line

There a 3 levels of verbose: 
- 'none' or None
- 'normal'
- 'high'
'''

import os
import time
from MIDAS.enums import Verbose
import inspect
from colorama import init as colorama_init, Fore, Back, Style

# Initialize colors
colorama_init()

# === REPORTER =============================================================

class cli_Reporter():

  def __init__(self, level=Verbose.NORMAL):

    self.level = level

    # Progress bar
    self.tref = None

  def __call__(self, s):

    self.send(s, Verbose.NORMAL)

  def normal(self, s):
    self.send(s, Verbose.NORMAL)

  def high(self, s):
    self.send(s, Verbose.HIGH)

  def line(self, text=None, thickness=1, char=None, color=Style.DIM, level=Verbose.NORMAL):
    '''Print a line spanning the whole command line.

    By default it is a single line (─) but other characters can be used: 
      Double line: ═
      Triple line: ≡
    '''

    # Terminal width
    try:
      tw = os.get_terminal_size().columns-10
    except:
      tw = 50

    # Thickness
    if char is None:
      match thickness:
        case 1: char = '─'
        case 2: char = '═'
        case 3: char = '≡'

    if text is None or text=='':
      S = color + char*tw + Style.RESET_ALL

    else:
      S = color + char*3 + Style.RESET_ALL + ' ' + text + ' '
      S += color + char*(tw-len(S)+len(color+Style.RESET_ALL)) + Style.RESET_ALL

    # Display
    self.send(S, level)

  def send(self, s, level):
    '''
    Send a verbose message to the CLI
    '''

    if level<=self.level:

      # --- Caller

      caller = inspect.stack()[2][0].f_locals
      if 'self' in caller:

        caller_type = caller['self'].__class__.__name__

        match caller_type:

          case 'Engine':
            print(Fore.CYAN + f'[Engine]  ' + Style.RESET_ALL, end='')

      else:
        print(Fore.RED + f'[Script] ' + Style.RESET_ALL, end='')

      # --- Verbose message
      
      print(s)
