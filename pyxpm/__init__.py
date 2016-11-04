#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''__init__
'''

import sys, os

def getConf(conf_file):
  return open(conf_file, 'rb').read().splitlines()

__conf__ = getConf(os.path.join(os.path.dirname(__file__), 'conf/setup.cf'))
__version__ = __conf__[0]
__url__ = 'https://github.com/sanjonemu/pyxpm'
__author__ = 'sanjonemu'
__author_email__ = 'sanjo_nemu@yahoo.co.jp'

XPM_CUSTOM = __conf__[2:]

from xpm import XPM, XPMINFOSIZE

__all__ = ['XPM', 'XPMINFOSIZE']
