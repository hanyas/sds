import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import stats

from sds.models import HiddenMarkovModel

import matplotlib.pyplot as plt

# npr.seed(1337)

true_hmm = HiddenMarkovModel(nb_states=3, obs_dim=2)

obs_dim = true_hmm.obs_dim
act_dim = true_hmm.act_dim
nb_states = true_hmm.nb_states

thetas = np.linspace(0, 2. * np.pi, nb_states, endpoint=False)
for k in range(true_hmm.nb_states):
    true_hmm.observations.sigma[k, ...] = sc.stats.invwishart.rvs(obs_dim + 1, np.eye(obs_dim))
    true_hmm.observations.mu[k, :] = 3. * np.array([np.cos(thetas[k]), np.sin(thetas[k])])

mat = 0.95 * np.eye(nb_states) + 0.05 * npr.rand(nb_states, nb_states)
mat /= np.sum(mat, axis=1, keepdims=True)
true_hmm.transitions.matrix = mat

# trajectory lengths
T = [95, 85, 75]

true_z, x = true_hmm.sample(horizon=T)
true_ll = true_hmm.log_normalizer(x)

npr.seed(1337)
ann_hmm = HiddenMarkovModel(nb_states=3, obs_dim=2)
ann_lls = ann_hmm.annealed_em(x, nb_iter=500, nb_sub_iter=25,
                              prec=0., discount=0.95)

npr.seed(1337)
std_hmm = HiddenMarkovModel(nb_states=3, obs_dim=2)
std_lls = std_hmm.em(x, nb_iter=500, prec=0., initialize=True)

print("true_ll=", true_ll, "std_ll=", std_lls[-1], "ann_ll=", ann_lls[-1])

plt.figure(figsize=(7, 7))
plt.axhline(y=true_ll, color='r')
plt.plot(ann_lls)
plt.plot(std_lls)
plt.xscale('symlog')
plt.yscale('symlog')
plt.show()

seq = npr.choice(len(x))
ann_hmm.plot(x[seq], true_state=true_z[seq], title='Annlead')
std_hmm.plot(x[seq], true_state=true_z[seq], title='Standard')
