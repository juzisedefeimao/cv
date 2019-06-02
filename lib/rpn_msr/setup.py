import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    Extension(
        "bbox",
        ["bbox.pyx"],
        # extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
	include_dirs = [numpy_include]
	)]

setup(
    name = "bbox pyx",
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)