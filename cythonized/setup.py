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
        "logsumexp_cy",
        ["logsumexp_cy.pyx"],
        extra_compile_args=['-ffast-math', '-O3', '-fopenmp'],
        extra_link_args=['-fopenmp'],
        libraries=["m"])
)

ext_modules.append(
    Extension(
        "hmm_cy",
        ["hmm_cy.pyx"],
        extra_compile_args=['-ffast-math', '-O3', '-fopenmp'],
        extra_link_args=['-fopenmp'],
        libraries=["m"])
)

ext_modules.append(
    Extension(
        "arhmm_cy",
        ["arhmm_cy.pyx"],
        extra_compile_args=['-ffast-math', '-O3', '-fopenmp'],
        extra_link_args=['-fopenmp'],
        libraries=["m"])
)

ext_modules.append(
    Extension(
        "rarhmm_cy",
        ["rarhmm_cy.pyx"],
        extra_compile_args=['-ffast-math', '-O3', '-fopenmp'],
        extra_link_args=['-fopenmp'],
        libraries=["m"])
)

ext_modules.append(
    Extension(
        "ararhmm_cy",
        ["ararhmm_cy.pyx"],
        extra_compile_args=['-ffast-math', '-O3', '-fopenmp'],
        extra_link_args=['-fopenmp'],
        libraries=["m"])
)

setup(
    include_dirs=[np.get_include()],
    ext_modules=cythonize(ext_modules),
)
