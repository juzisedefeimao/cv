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
        "cython_nms",
        ["nms.pyx"],
        # extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
	include_dirs = [numpy_include]
	)]

setup(
    name = "bbox pyx",
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)





# ext_modules = [
#     Extension(
#         "cython_nms",
#         ["nms.pyx"],
#         # extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
#         include_dirs=[numpy_include]
#     )]
    # Extension(
    #     "cpu_nms",
    #     ["cpu_nms.pyx"],
    #     # extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
    #     include_dirs=[numpy_include]
    # ),
    # Extension(
    #     "gpu_nms",
    #     ["gpu_nms.pyx"],
    #     # language='c',
    #     # extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
    #     include_dirs=[numpy_include, 'C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include']
    # )

# setup(
#     name = "nms",
#     ext_modules = ext_modules,
#     cmdclass = {'build_ext': build_ext}
# )
