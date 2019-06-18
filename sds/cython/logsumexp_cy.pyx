#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: speed
# @Date: 2019-06-05-19-23
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import cython
cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

from libc.math cimport log, exp, fmax, INFINITY

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)

cpdef double logsumexp1d(double[:] x) nogil:
    cdef int i, N
    cdef double m, out

    N = x.shape[0]

    # find the max
    m = -INFINITY
    for i in range(N):
        m = fmax(m, x[i])

    # sum the exponentials
    out = 0
    for i in range(N):
        out += exp(x[i] - m)

    return m + log(out)


cpdef np.ndarray[DTYPE_t, ndim=1] logsumexp2d(double[:, :] x):
    cdef int i

    cdef N = x.shape[0]
    cdef D = x.shape[1]

    cdef double[:] mat
    mat = np.empty((N, ))

    for i in range(N):
        mat[i] = logsumexp1d(x[i, :])

    return np.asarray(mat)