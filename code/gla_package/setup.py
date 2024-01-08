# several files with ext .pyx, that i will call by their name
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules=[
    Extension("gla",       ["gla.pyx"]),
    Extension("landscape",         ["landscape.pyx"]),
    Extension("fit",  ["fit.pyx"]),
]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"}


setup(
  name = 'gla_package',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
  include_dirs=["."],
)