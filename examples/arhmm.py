import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import stats

from sds.models import ARHMM
from sds.utils.general import random_rotation

import matplotlib.pyplot as plt

# npr.seed(1337)

true_arhmm = ARHMM(nb_states=5, obs_dim=2, algo_type='ML')

obs_dim = true_arhmm.obs_dim
act_dim = true_arhmm.act_dim
nb_states = true_arhmm.nb_states

for k in range(nb_states):
    true_arhmm.observations.sigma[k, ...] = sc.stats.invwishart.rvs(obs_dim + 1, np.eye(obs_dim))
    true_arhmm.observations.A[k, ...] = .95 * random_rotation(obs_dim)
    true_arhmm.observations.B[k, ...] = npr.randn(obs_dim, act_dim)
    true_arhmm.observations.c[k, :] = npr.randn(obs_dim)

mat = 0.95 * np.eye(nb_states) + 0.05 * npr.rand(nb_states, nb_states)
mat /= np.sum(mat, axis=1, keepdims=True)
true_arhmm.transitions.matrix = mat

# trajectory lengths
T = [150, 135, 165]

true_z, x = true_arhmm.sample(horizon=T)
true_ll = true_arhmm.log_normalizer(x)

npr.seed(1337)
ann_arhmm = ARHMM(nb_states=5, obs_dim=2)
ann_lls = ann_arhmm.annealed_em(x, nb_iter=1000,
                                prec=0., discount=0.99)

npr.seed(1337)
std_arhmm = ARHMM(nb_states=5, obs_dim=2)
std_lls = std_arhmm.em(x, nb_iter=500, prec=0., initialize=True)

print("true_ll=", true_ll, "std_ll=", std_lls[-1], "ann_ll=", ann_lls[-1])

plt.figure(figsize=(7, 7))
plt.axhline(y=true_ll, color='r')
plt.plot(ann_lls)
plt.plot(std_lls)
plt.xscale('symlog')
plt.yscale('symlog')
plt.show()

seq = npr.choice(len(x))
ann_arhmm.plot(x[seq], true_state=true_z[seq], title='Annlead')
std_arhmm.plot(x[seq], true_state=true_z[seq], title='Standard')
