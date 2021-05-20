# python setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


ext_modules = []

ext_modules.append(
    Extension('sds.cython.hmm_cy',
              sources=["sds/cython/hmm_cy.pyx"],
              language="c++")
)


setup(name='sds',
      version='0.0.1',
      description='Switching dynamical systems for control',
      author='Hany Abdulsamad',
      author_email='hany@robot-learning.de',
      install_requires=['numpy', 'scipy', 'matplotlib',
                        'seaborn', 'sklearn', 'tqdm',
                        'pathos', 'torch', 'cython', 'gym'],
      packages=['sds'], ext_modules=cythonize(ext_modules),
      include_dirs=[np.get_include()],)
