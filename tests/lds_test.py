import numpy as np
import numpy.random as npr
from numpy.random import multivariate_normal as mvn

np.set_printoptions(precision=3)

npr.seed(1337)

# Example of a car from Särkkä
# Bayesian Filtering and Smoothing
dt = 0.1
q1, q2 = 1., 1.
s1, s2 = .25, .25

A = np.array([[1., 0., dt, 0.],
              [0., 1., 0., dt],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]])

c = np.zeros((4, ))

Q = np.array([[q1 * dt**3 / 3., 0., q1 * dt**2 / 2., 0.],
              [0., q2 * dt**3 / 3., 0., q2 * dt**2 / 2.],
              [q1 * dt**2 / 2., 0., q1 * dt, 0.],
              [0., q2 * dt**2 / 2., 0., q2 * dt]])

H = np.array([[1., 0., 0., 0.],
              [0., 1., 0., 0.]])

g = np.zeros((2, ))

R = np.array([[s1**2, 0.],
              [0., s2**2]])

T = 250

x0 = np.array([0., 0., 1., 1.])
S0 = 0.01 * np.eye(4)

X, Y = [], []

for _ in range(5):
    x = np.zeros((T, 4))
    y = np.zeros((T, 2))

    x[0] = x0 + mvn(mean=np.zeros((4,)), cov=S0)
    y[0] = H @ x[0] + g + mvn(mean=np.zeros((2,)), cov=R)

    for t in range(1, T):
        x[t] = A @ x[t - 1] + c + mvn(mean=np.zeros((4,)), cov=Q)
        y[t] = H @ x[t] + g + mvn(mean=np.zeros((2,)), cov=R)

    X.append(x)
    Y.append(y)


def gaussian_logpdf(y, m, S):

    dim = m.shape[0]
    L = np.linalg.cholesky(S)
    x = np.linalg.solve(L, y - m)
    return - 0.5 * dim * np.log(2. * np.pi)\
           - np.sum(np.log(np.diag(L))) - 0.5 * np.sum(x**2)


def kalman_filter(m0, S0,  # prior
                  A, c, Q,  # dynamics
                  H, g, R,  # observation
                  y):  # data

    T = y.shape[0]
    dy = y.shape[1]
    dx = m0.shape[0]

    log_lik = 0.

    mf = np.zeros((T, dx))
    Sf = np.zeros((T, dx, dx))

    mp = m0
    Sp = S0

    for t in range(T):
        # symmetrize
        Sp = 0.5 * (Sp + Sp.T)

        # log_lik
        log_lik += gaussian_logpdf(y[t], H @ mp + g,
                                  H @ Sp @ H.T + R)

        # condition
        K = np.linalg.solve(R + H @ Sp @ H.T, H @ Sp).T

        mf[t] = mp + K @ (y[t] - (H @ mp + g))
        Sf[t] = Sp - K @ (H @ Sp @ H.T + R) @ K.T
        Sf[t] = 0.5 * (Sf[t] + Sf[t].T)

        # predict
        mp = A @ mf[t] + c
        Sp = A @ Sf[t] @ A.T + Q

    return mf, Sf, log_lik


# RTS-Smoother
def kalman_smoother(m0, S0,
                    A, c, Q,
                    H, g, R,
                    y, Es=False):

    mf, Sf, ll = kalman_filter(m0, S0,
                               A, c, Q,
                               H, g, R,
                               y)

    T = y.shape[0]
    dy = y.shape[1]
    dx = m0.shape[0]

    ms = np.zeros((T, dx))
    Ss = np.zeros((T, dx, dx))

    ms[-1] = mf[-1]
    Ss[-1] = Sf[-1]

    G = np.zeros((T - 1, dx, dx))

    for t in range(T - 2, -1, -1):
        G[t] = np.linalg.solve(Q + A @ Sf[t] @ A.T, A @ Sf[t]).T
        ms[t] = mf[t] + G[t] @ (ms[t + 1] - (A @ mf[t] + c))
        Ss[t] = Sf[t] + G[t] @ (Ss[t + 1] - A @ Sf[t] @ A.T - Q) @ G[t].T

    if Es:
        Ex = ms        # E[x{n}]
        ExxpT = np.zeros_like(G)     # E[x_{n} x_{n-1}^T]
        for t in range(T - 1):
            ExxpT[t] = Ss[t + 1] @ G[t].T + np.outer(ms[t + 1], ms[t])

        ExxT = np.zeros_like(Ss)   # E[x_{n} x_{n}^T]
        for t in range(T):
            ExxT[t] = Ss[t] + np.outer(ms[t], ms[t])

        return ms, Ss, ll, tuple([Ex, ExxpT, ExxT])
    else:
        return ms, Ss, ll


def estep(m0, S0,
          A, c, Q,
          H, g, R,
          y):

    ms, Ss, ll, Es = kalman_smoother(m0, S0,
                                     A, c, Q,
                                     H, g, R,
                                     y, Es=True)

    Ex, ExxpT, ExxT = Es

    return ms, Ss, ll, Ex, ExxpT, ExxT


import matplotlib.pyplot as plt

ll = 0.
for x, y in zip(X, Y):
    mf, Sf, _ll = kalman_filter(np.zeros((4, )),
                                1e8 * np.eye(4),
                                A, c, Q,
                                H, g, R,
                                y)

    ms, Ss = kalman_smoother(np.zeros((4, )),
                             1e8 * np.eye(4),
                             A, c, Q,
                             H, g, R,
                             y)[0:2]

    ll += _ll

    plt.figure()

    plt.plot(x[:, 0], x[:, 1], 'red', label='true')
    plt.scatter(y[:, 0], y[:, 1], marker='+', label='obs')
    plt.plot(mf[:, 0], mf[:, 1], 'green', label='filter')
    plt.plot(ms[:, 0], ms[:, 1], 'darkviolet', label='smoother')

    plt.legend(loc="upper left")
    plt.show()

    ferr = np.linalg.norm(mf - x)
    serr = np.linalg.norm(ms - x)
    print("Filter error: ", ferr,
          "Smoother error: ", serr)

print("Loglik:", ll)

# x0_, S0_ = x0, S0
# A_, c_, Q_ = A, c, Q
H_, g_, R_ = H, g, R

x0_ = np.zeros((4, ))
S0_ = np.eye(4) * 1e8

A_ = npr.randn(4, 4)
c_ = npr.randn(4)
Q_ = np.eye(4)

# H_ = npr.randn(2, 4)
# g_ = npr.randn(2)
# R_ = np.eye(2)

N = len(Y)

lls = []
for it in range(250):
    ll = 0.
    Ex, ExxpT, ExxT = [], [], []
    for y in Y:
        _, _, _ll, _Ex, _ExxpT, _ExxT = estep(x0_, S0_,
                                              A_, c_, Q_,
                                              H_, g_, R_,
                                              y)

        ll += _ll

        Ex.append(_Ex)
        ExxpT.append(_ExxpT)
        ExxT.append(_ExxT)

    lls.append(ll)

    # --------------------------------------------------------- #

    iEx = np.mean(np.array(Ex), axis=0)
    iExxT = np.mean(np.array(ExxT), axis=0)

    x0_ = iEx[0]
    S0_ = iExxT[0] - np.outer(iEx[0], iEx[0])

    # --------------------------------------------------------- #

    eEx = np.mean(np.array(Ex), axis=0)
    eExxpT = np.mean(np.array(ExxpT), axis=0)
    eExxT = np.mean(np.array(ExxT), axis=0)

    xxT = np.zeros((T - 1, 4 + 1, 4 + 1))
    for t in range(T - 1):
        xxT[t] = np.block([[eExxT[t], eEx[t][:, np.newaxis]],
                           [eEx[t][np.newaxis, :], np.ones((1,))]])

    yxT = np.zeros((T - 1, 4, 4 + 1))
    for t in range(T - 1):
        yxT[t] = np.hstack((eExxpT[t], eEx[t + 1][:, np.newaxis]))

    Ac = np.linalg.solve(np.sum(xxT, axis=0).T, np.sum(yxT, axis=0).T).T

    A_, c_ = Ac[:, :4], Ac[:, -1]

    yyT = np.sum(eExxT[1:], axis=0)
    xpxTAT = np.einsum('nij,kj->ik', yxT, Ac)
    AxxTAT = np.einsum('ij,njk,lk->il', Ac, xxT, Ac)

    Q_ = (yyT - 2. * xpxTAT + AxxTAT) / (T - 1)

    # --------------------------------------------------------- #

    # vEx = np.vstack(Ex)
    # vExxT = np.vstack(ExxT)
    # vy = np.vstack(Y)
    #
    # NT = len(vy)
    #
    # xxT = np.zeros((NT, 4 + 1, 4 + 1))
    # for t in range(NT):
    #     xxT[t] = np.block([[vExxT[t], vEx[t][:, np.newaxis]],
    #                        [vEx[t][np.newaxis, :], np.ones((1,))]])
    #
    # x = np.hstack((vEx, np.ones((vEx.shape[0], 1))))
    # Hg = np.linalg.solve(np.sum(xxT, axis=0).T, np.einsum('nd,nl->dl', vy, x).T).T
    #
    # H_, g_ = Hg[:, :4], Hg[:, -1]
    #
    # yyT = np.einsum('ni,nj->ij', vy, vy)
    # HxyT = np.einsum('ij,nj,nl->il', Hg, x, vy)
    # HxxTHT = np.einsum('ij,njk,lk->il', Hg, xxT, Hg)
    #
    # R_ = (yyT - 2 * HxyT + HxxTHT) / NT

    # --------------------------------------------------------- #

    print('it:', it, 'll:', ll)

print("ll monotonic?:", np.all(np.diff(lls) >= -1e-8))

import matplotlib.pyplot as plt

x, y = X[0], Y[0]

mf, Sf, ll = kalman_filter(x0_, S0_,
                           A_, c_, Q_,
                           H_, g_, R_,
                           y)

ms, Ss = kalman_smoother(x0_, S0_,
                         A_, c_, Q_,
                         H_, g_, R_,
                         y)[0:2]

plt.figure()

plt.plot(x[:, 0], x[:, 1], 'red', label='true')
plt.scatter(y[:, 0], y[:, 1], marker='+', label='obs')
plt.plot(mf[:, 0], mf[:, 1], 'green', label='filter')
plt.plot(ms[:, 0], ms[:, 1], 'darkviolet', label='smoother')

plt.legend(loc="upper left")
plt.show()

ferr = np.linalg.norm(mf - x)
serr = np.linalg.norm(ms - x)
print("Filter error: ", ferr,
      "Smoother error: ", serr)

print("Loglik:", ll)
