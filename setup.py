import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_module = Extension(name="gait2dpi_dGRF",
                       sources=["gait2dpi_dGRF.pyx",
                                "gait2dpi_dGRF_org.c"],
                       include_dirs=[numpy.get_include()])

setup(name="gait2dpi_dGRF",
      cmdclass = {'build_ext': build_ext},
      ext_modules = [ext_module])
