#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''xpm
dummy checker 7x5_c16
FF0000 FFFFFF FF0000 FFFFFF FF0000 FFFFFF FF0000
FFFFFF 00FF00 FFFFFF 00FF00 FFFFFF 00FF00 FFFFFF
0000FF FFFFFF 0000FF FFFFFF 0000FF FFFFFF 0000FF
FFFFFF 333333 FFFFFF 666666 FFFFFF CCCCCC FFFFFF
00FFFF FFFFFF FF00FF FFFFFF FFFF00 FFFFFF 000000
'''

import sys
import numpy as np

bm = np.array([
[[1.,0.,0.],[1.,1.,1.],[1.,0.,0.],[1.,1.,1.],[1.,0.,0.],[1.,1.,1.],[1.,0.,0.]],
[[1.,1.,1.],[0.,1.,0.],[1.,1.,1.],[0.,1.,0.],[1.,1.,1.],[0.,1.,0.],[1.,1.,1.]],
[[0.,0.,1.],[1.,1.,1.],[0.,0.,1.],[1.,1.,1.],[0.,0.,1.],[1.,1.,1.],[0.,0.,1.]],
[[1.,1.,1.],[.2,.2,.2],[1.,1.,1.],[.4,.4,.4],[1.,1.,1.],[.8,.8,.8],[1.,1.,1.]],
[[0.,1.,1.],[1.,1.,1.],[1.,0.,1.],[1.,1.,1.],[1.,1.,0.],[1.,1.,1.],[0.,0.,0.]]
], dtype=np.float32)

def XPM(*args, **kwargs):
  sys.stderr.write('''** remove 'xpm.py' and replace to 'xpm.pyd' later **''')
  return bm
