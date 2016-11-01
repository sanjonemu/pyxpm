from distutils.core import setup
import os

mdl = __import__('pyxpm')

kwargs = {
  'name': 'pyxpm',
  'version': mdl.__version__,
  'keywords': 'XPM numpy PIL image',
  'description': ('XPM image file loader for Python (to numpy ndarray or PIL) native C .pyd'),
  'long_description': open('README.md', 'rb').read(),
  'author': mdl.__author__,
  'author_email': mdl.__author_email__,
  'url': mdl.__url__,
  'packages': ['pyxpm'],
  'package_dir': {'pyxpm': './pyxpm'},
  'package_data': {'pyxpm': [
    'conf/setup.cf',
    'include/xpm.h',
    'src/xpm.c',
    'src/makefile.tdmgcc64',
    'cs_Tux_58x64_c16.xpm',
    'cs_Tux_ecb_58x64_c16.xpm']},
  'requires': ['numpy', 'PIL'],
  'classifiers': [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: System Administrators',
    'Intended Audience :: End Users/Desktop',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 2.5',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 2 :: Only',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Utilities'],
  'license': 'MIT License'}

setup(**kwargs)
