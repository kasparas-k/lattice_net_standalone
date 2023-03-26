import pathlib
from setuptools import setup, Extension
from setuptools import setup
from distutils.sysconfig import get_python_inc

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PROJ_DIR = pathlib.Path(__file__).parent.resolve()
EIGEN_INCLUDE = '/usr/include/eigen3'
OPENCV_INCLUDE = '/usr/include/opencv4'

setup(
    name='latticenet',
    version='1.0.0',
    author="Radu Alexandru Rosu",
    author_email="rosu@ais.uni-bonn.de",
    description="LatticeNet",
    zip_safe=False,
    ext_modules=[
        CUDAExtension(
            name='latticenet',
            sources=['src/Lattice.cu',
                     'src/HashTable.cu',
                     'src/PyBridge.cxx'],
            include_dirs=[str(PROJ_DIR / 'include'),
                          str(PROJ_DIR / 'deps/include'),
                          str(PROJ_DIR / 'deps'),
                          OPENCV_INCLUDE,
                          EIGEN_INCLUDE],
            extra_compile_args=[f'-D PROJECT_SOURCE_DIR="{PROJ_DIR}"',
                                f'-D CMAKE_SOURCE_DIR="{PROJ_DIR}"'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
