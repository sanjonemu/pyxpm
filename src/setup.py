#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''setup
'''

from distutils.core import setup, Extension
# from numpy.distutils.misc_util import get_numpy_include_dirs
import numpy

xpm = Extension(
  'xpm',
  sources=['xpm.c'],
  include_dirs=[numpy.get_include()], # + get_numpy_include_dirs,
  libraries=[],
  library_dirs=[],
  extra_compile_args=[],
  extra_link_args=[])

kwargs = {
  # 'packages': [], # [pkgname, pkgname, ...]
  # 'package_dir': Dict(), # {pkgname: path, pkgname: path, ...}
  # 'package_data': Dict(), # {pkgname: [path, path, ...], pkgname: [...], ...}
  # 'data_files': [], # [(path, [pat, path, ...]), (path, [...]), ...]
  # 'scripts': [], # [path, path, ...]
  'ext_modules': [xpm]
}

setup(**kwargs)
