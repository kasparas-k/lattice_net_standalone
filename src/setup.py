from setuptools import setup, Extension
from setuptools import setup
from distutils.sysconfig import get_python_inc

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
            sources=['Lattice.cu',
                     'HashTable.cu',
                     'PyBridge.cxx'],
            include_dirs=['/home/kasparas/Documents/pointcloud_nn/algorithms/myfork/lattice_net_standalone/include',
                                '/home/kasparas/Documents/pointcloud_nn/algorithms/myfork/lattice_net_standalone/deps/include',
                                '/home/kasparas/Documents/pointcloud_nn/algorithms/myfork/lattice_net_standalone/deps/',
                                '/usr/include/eigen3',
                                '/usr/include/opencv4'],
            extra_compile_args=['-D PROJECT_SOURCE_DIR="/home/kasparas/Documents/pointcloud_nn/algorithms/myfork/lattice_net_standalone"',
                                '-D CMAKE_SOURCE_DIR="/home/kasparas/Documents/pointcloud_nn/algorithms/myfork/lattice_net_standalone"'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
