# python setup.py build_ext --inplace

import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

USE_OPENMP = os.environ.get('USE_OPENMP', False)
print("USE_OPENMP", USE_OPENMP)

ext_modules = []

# ext_modules.append(
#     Extension('sds_numpy.cython.hmm_cy',
#               sources=["sds_numpy/cython/hmm_cy.pyx"],
#               language="c++")
# )

ext_modules.append(
    Extension('sds_numpy.cython.hmm_cy',
              extra_compile_args=["-fopenmp"] if USE_OPENMP else ["-stdlib=libc++"],
              extra_link_args=["-fopenmp"] if USE_OPENMP else ["-stdlib=libc++"],
              sources=["sds_numpy/cython/hmm_cy.pyx"],
              include_dirs = ['sds_numpy/cython/'],
              language="c++")
)


setup(name='sds_numpy',
      version='0.0.1',
      description='Switching dynamical systems for control',
      author='Hany Abdulsamad',
      author_email='hany@robot-learning.de',
      install_requires=['numpy', 'scipy', 'matplotlib', 'sklearn'],
      packages=['sds_numpy'], ext_modules=cythonize(ext_modules),
      include_dirs=[np.get_include()],)
