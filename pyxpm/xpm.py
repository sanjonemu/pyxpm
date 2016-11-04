#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''xpm
dummy checker 7x5_c16
'''

import sys
import numpy as np

F, C, H, G, D, O = 255, 203, 159, 101, 50, 0

bm_RGBA = np.array([ # RGB(A)
 [[F,O,O,F], [F,F,F,F], [F,O,O,F], [F,F,F,F], [F,O,O,H], [F,F,F,F], [F,O,O,F]],
 [[F,F,F,F], [O,F,O,F], [F,F,F,F], [O,F,O,H], [F,F,F,F], [O,F,O,F], [F,F,F,F]],
 [[O,O,F,F], [F,F,F,F], [O,O,F,H], [F,F,F,F], [O,O,F,F], [F,F,F,F], [O,O,F,F]],
 [[F,F,F,F], [D,D,D,F], [F,F,F,F], [G,G,G,F], [F,F,F,F], [C,C,C,F], [F,F,F,F]],
 [[O,F,F,F], [F,F,F,F], [F,O,F,F], [F,F,F,F], [F,F,O,F], [F,F,F,F], [O,O,O,F]]
], dtype=np.uint8)

bm_BGRA = np.array([ # BGR(A)
 [[O,O,F,F], [F,F,F,F], [O,O,F,F], [F,F,F,F], [O,O,F,H], [F,F,F,F], [O,O,F,F]],
 [[F,F,F,F], [O,F,O,F], [F,F,F,F], [O,F,O,H], [F,F,F,F], [O,F,O,F], [F,F,F,F]],
 [[F,O,O,F], [F,F,F,F], [F,O,O,H], [F,F,F,F], [F,O,O,F], [F,F,F,F], [F,O,O,F]],
 [[F,F,F,F], [D,D,D,F], [F,F,F,F], [G,G,G,F], [F,F,F,F], [C,C,C,F], [F,F,F,F]],
 [[F,F,O,F], [F,F,F,F], [F,O,F,F], [F,F,F,F], [O,F,F,F], [F,F,F,F], [O,O,O,F]]
], dtype=np.uint8)

def XPM(*args, **kwargs):
  sys.stderr.write('''** remove 'xpm.py' and replace to 'xpm.pyd' later **''')
  # return bm_RGBA
  return bm_BGRA

def XPMINFOSIZE():
  return 0
