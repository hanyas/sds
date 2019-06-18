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

cdef double logsumexp(double[::1] x) nogil:
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


cpdef filter_cy(double[::1] loginit,
                double[:,::1] logtrans,
                double[:,::1] logobs,
                double[:,::1] alpha):

    cdef int T, K, t, k, j
    T = logobs.shape[0]
    K = logobs.shape[1]

    for k in range(K):
        alpha[0, k] = loginit[k] + logobs[0, k]

    cdef double[::1] aux = np.zeros(K)
    for t in range(1, T):
        for k in range(K):
            for j in range(K):
                aux[j] = alpha[t - 1, j] + logtrans[j, k]
            alpha[t, k] = logsumexp(aux) + logobs[t, k]


cpdef smooth_cy(double[::1] loginit,
                double[:,::1] logtrans,
                double[:,::1] logobs,
                double[:,::1] beta):

    cdef int T, K, t, k, j
    T = logobs.shape[0]
    K = logobs.shape[1]

    for k in range(K):
        beta[T - 1, k] = 0.0

    cdef double[::1] aux = np.zeros(K)
    for t in range(T - 2, -1, -1):
        for k in range(K):
            for j in range(K):
                aux[j] = logtrans[k, j] + beta[t + 1, j] + logobs[t + 1, j]
            beta[t, k] = logsumexp(aux)
