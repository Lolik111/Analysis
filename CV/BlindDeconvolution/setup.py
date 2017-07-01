
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension("utils", ["utils.pyx"], include_dirs=[numpy.get_include()]
    ),
]

setup(
    name="BlindDeconvolution",
    version="0.0.1",
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    packages=["CV.BlindDeconvolution"],
)