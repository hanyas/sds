import numpy as np
import numpy.random as npr

from numpy.random import multivariate_normal as mvn

np.set_printoptions(precision=3)

# npr.seed(1337)

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

g = np.array([0., 0.])

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


from sds.models import LinearGaussianDynamics

ltn_dim = 4
ltn_lag = 1
ems_dim = 2
act_dim = 0

# init_ltn_prior
mu = np.zeros((ltn_dim,))
kappa = 1e-2
psi = np.eye(ltn_dim)
nu = (ltn_dim + 1) + 1e-2

from sds.distributions.composite import NormalWishart
init_ltn_prior = NormalWishart(ltn_dim, mu, kappa, psi, nu)

# ltn_prior
input_dim = ltn_dim * ltn_lag + act_dim + 1
output_dim = ltn_dim

M = np.zeros((output_dim, input_dim))
K = 1e-2 * np.eye(input_dim)
psi = np.eye(output_dim)
nu = (output_dim + 1) + 1e-8

from sds.distributions.composite import MatrixNormalWishart

ltn_prior = MatrixNormalWishart(input_dim, output_dim,
                                M, K, psi, nu)

# emission_prior
input_dim = ltn_dim + 1
output_dim = ems_dim

M = np.zeros((output_dim, input_dim))
K = 1e-2 * np.eye(input_dim)
psi = np.eye(output_dim)
nu = (output_dim + 1) + 1e-8

from sds.distributions.composite import MatrixNormalWishart

ems_prior = MatrixNormalWishart(input_dim, output_dim,
                                M, K, psi, nu)

# model kwargs
lds = LinearGaussianDynamics(ems_dim=ems_dim, act_dim=act_dim,
                             ltn_dim=ltn_dim, ltn_lag=ltn_lag,
                             init_ltn_prior=init_ltn_prior,
                             ltn_prior=ltn_prior,
                             ems_prior=ems_prior)

# lds.init_latent.params = x0, np.linalg.inv(S0)
# lds.latent.params = np.column_stack([A, c]), np.linalg.inv(Q)
lds.emission.params = np.column_stack([H, g]), np.linalg.inv(R)

lls = lds.em(Y, nb_iter=250, tol=0.)
print("ll monotonic?", np.all(np.diff(lls) >= 0.))
