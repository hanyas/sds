#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: setup.py
# @Date: 2019-06-05-19-15
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

# python setup.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


ext_modules = []

ext_modules.append(
    Extension(
        'sds.cython.logsumexp_cy',
        sources=["sds/cython/logsumexp_cy.pyx"],
        language="c++")
)

ext_modules.append(
    Extension(
        'sds.cython.hmm_cy',
        sources=["sds/cython/hmm_cy.pyx"],
        language="c++")
)


setup(name='sds',
      version='0.0.1',
      description='Switching dynamical systems for control',
      author='Hany Abdulsamad',
      author_email='hany@robot-learning.de',
      install_requires=['numpy', 'scipy', 'matplotlib', 'autograd'],
      packages=['sds'], ext_modules=cythonize(ext_modules),
      include_dirs=[np.get_include()],)
