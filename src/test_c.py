#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''test_c
'''

import sys, os

import ctypes

XPM_PYD = '../dll/xpm.pyd'
XPM_FILE = '/tmp/testdata.xpm'

def main():
  xpm = ctypes.cdll.LoadLibrary(XPM_PYD)
  sys.stderr.write('xpminfosize: %d\n' % xpm.getxpminfosize())
  xi = ctypes.create_string_buffer(xpm.getxpminfosize())
  s = ctypes.create_string_buffer(open(XPM_FILE, 'rb').read())
  sys.stderr.write('xpm: %d\n' % xpm.loadxpm(xi, s))
  p = ctypes.cast(xi, ctypes.POINTER(ctypes.c_int))
  sys.stderr.write('(%d, %d, %d): %d\n' % (p[1], p[0], p[8], p[9]))
  q = ctypes.cast(xi, ctypes.POINTER(ctypes.c_ubyte))
  o = 12 * 4 + 256 * 80
  a = (q[o+3] << 24) | (q[o+2] << 16) | (q[o+1] << 8) | q[o]
  sys.stderr.write('%08X\n' % a)
  if a == 0: return False
  sys.stderr.write('[\n')
  b = ctypes.cast(a, ctypes.POINTER(ctypes.c_uint))
  for y in range(p[1]):
    sys.stderr.write('%04X:[\n' % y)
    for x in range(p[0]):
      sys.stderr.write('#%08X ' % b[y * p[0] + x])
    sys.stderr.write('],\n')
  sys.stderr.write(']\n')
  xpm.freexpm(xi)
  a = (q[o+3] << 24) | (q[o+2] << 16) | (q[o+1] << 8) | q[o]
  sys.stderr.write('%08X\n' % a)
  return True

if __name__ == '__main__':
  main()
