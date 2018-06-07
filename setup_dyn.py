import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_module = Extension(name="gait2dp_dyn",
                       sources=["gait2dp_dyn.pyx",
                                "gait2dp_dyn.c"],
                       include_dirs=[numpy.get_include()])

setup(name="gait2dp_dyn",
      cmdclass = {'build_ext': build_ext},
      ext_modules = [ext_module])
