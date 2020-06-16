import numpy as np
cimport numpy as np

from libc.math cimport log, exp, fmax, INFINITY


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


cpdef forward_cy(double[::1] loginit,
                 double[:,:,::1] logtrans,
                 double[:,::1] logobs,
                 double[:,::1] logctl,
                 double[:,::1] alpha,
                 double[::1] norm):

    cdef int T, K, t, k, j
    T = logobs.shape[0]
    K = logobs.shape[1]

    for k in range(K):
        alpha[0, k] = loginit[k] + logobs[0, k]

    norm[0] = logsumexp(alpha[0])
    for k in range(K):
        alpha[0, k] = alpha[0, k] - norm[0]

    cdef double[::1] aux = np.zeros(K)
    for t in range(1, T):
        for k in range(K):
            for j in range(K):
                aux[j] = alpha[t - 1, j] + logtrans[t - 1, j, k]
            alpha[t, k] = logsumexp(aux) + logobs[t, k] + logctl[t, k]

        norm[t] = logsumexp(alpha[t])
        for k in range(K):
            alpha[t, k] = alpha[t, k] - norm[t]

cpdef backward_cy(double[::1] loginit,
                  double[:,:,::1] logtrans,
                  double[:,::1] logobs,
                  double[:,::1] logctl,
                  double[:,::1] beta,
                  double[::1] scale):

    cdef int T, K, t, k, j
    T = logobs.shape[0]
    K = logobs.shape[1]

    for k in range(K):
        beta[T - 1, k] = 0.0 - scale[T - 1]

    cdef double[::1] aux = np.zeros(K)
    for t in range(T - 2, -1, -1):
        for k in range(K):
            for j in range(K):
                aux[j] = logtrans[t, k, j] + beta[t + 1, j]\
                         + logobs[t + 1, j] + logctl[t + 1, j]
            beta[t, k] = logsumexp(aux) - scale[t]
