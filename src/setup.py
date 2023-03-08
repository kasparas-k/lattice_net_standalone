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
                     'HashTable.cu'],
            extra_include_dirs=['/home/kasparas/Documents/pointcloud_nn/algorithms/myfork/lattice_net_standalone/include/lattice_net'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
