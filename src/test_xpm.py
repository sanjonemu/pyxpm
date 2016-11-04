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
from PIL import Image

sys.path.append('..')
from pyxpm import xpm

XPM_BASE = '/tmp'
XPM_INFILE = '%s/testdata.xpm' % XPM_BASE
XPM_OUTGIF_0 = '%s/testdata_0.gif' % XPM_BASE
XPM_OUTGIF_1 = '%s/testdata_1.gif' % XPM_BASE
XPM_OUTPNG = '%s/testdata.png' % XPM_BASE

def main():
  fig = plt.figure()
  axis = [fig.add_subplot(211 + _) for _ in range(2)]
  s = open(XPM_INFILE, 'rb').read()
  nda = xpm.XPM(s) # as ndarray (dtype=np.uint8) BGR(A)
  sys.stderr.write('%s\n' % str(nda))
  r, c, m = nda.shape
  img = Image.frombuffer('RGBA', (c, r), nda, 'raw', 'BGRA', 0, 1)
  img.show() # PIL.Image
  bm = np.array(img) # RGB(A)
  axis[0].imshow(bm)
  # misc.imsave(XPM_OUTGIF_0, np.uint8(bm)) # changed
  misc.imsave(XPM_OUTGIF_0, np.float32(bm)) # color changed
  # misc.imsave(XPM_OUTGIF_1, misc.bytescale(bm, cmin=0, cmax=255)) # changed
  misc.imsave(XPM_OUTGIF_1, misc.bytescale(bm)) # color changed
  # im = misc.toimage(bm, cmin=0, cmax=255) # same as PIL.Image
  im = misc.toimage(bm) # same as PIL.Image
  im.save(XPM_OUTPNG) # palette ok
  axis[1].imshow(im)
  plt.show()

if __name__ == '__main__':
  main()
