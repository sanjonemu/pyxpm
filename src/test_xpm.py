#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''test_xpm
.xpm((byte)string) -> xpm.XPM() -> ndarray(numpy) -> image(numpy/PIL)
'''

import sys, os

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from scipy import misc
# from PIL import Image

sys.path.append('../dll')
import xpm

XPM_BASE = '/tmp'
XPM_INFILE = '%s/testdata.xpm' % XPM_BASE
XPM_OUTGIF_0 = '%s/testdata_0.gif' % XPM_BASE
XPM_OUTGIF_1 = '%s/testdata_1.gif' % XPM_BASE
XPM_OUTPNG = '%s/testdata.png' % XPM_BASE

def main():
  sys.stderr.write('[%s]\n' % xpm.XPM(XPM_INFILE))
  return

  fig = plt.figure()
  axis = [fig.add_subplot(121 + _) for _ in range(2)]
  bm = xpm.XPM(XPM_INFILE) # as ndarray
  axis[0].imshow(bm)
  misc.imsave(XPM_OUTGIF_0, np.uint8(bm))
  misc.imsave(XPM_OUTGIF_1, misc.bytescale(bm, cmin=0, cmax=255))
  im = misc.toimage(bm, cmin=0, cmax=255) # same as PIL.Image
  axis[1].imshow(im)
  im.save(XPM_OUTPNG)
  plt.show()

if __name__ == '__main__':
  main()
