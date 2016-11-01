#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''setup
'''

from distutils.core import setup, Extension
import numpy

# define the extension module
xpm = Extension('xpm', sources=['xpm.c'],
  include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[xpm])
