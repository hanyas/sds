import autograd.numpy as np
import autograd.numpy.random as npr

from sds import HMM
from sds.utils import permutation

import matplotlib.pyplot as plt
from hips.plotting.colormaps import gradient_cmap

import seaborn as sns


sns.set_style("white")
sns.set_context("talk")

color_names = ["windows blue", "red", "amber", "faded green", "dusty purple", "orange"]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

true_hmm = HMM(nb_states=3, dm_obs=2)

thetas = np.linspace(0, 2 * np.pi, true_hmm.nb_states, endpoint=False)
for k in range(true_hmm.nb_states):
    true_hmm.observations.mu[k, :] = 3 * np.array([np.cos(thetas[k]), np.sin(thetas[k])])

# trajectory lengths
T = [95, 85, 75]

true_z, x = true_hmm.sample(horizon=T)
true_ll = true_hmm.log_probability(x)

hmm = HMM(nb_states=3, dm_obs=2)
hmm.initialize(x)

lls = hmm.em(x, nb_iter=1000, prec=0., verbose=True)
print("true_ll=", true_ll, "hmm_ll=", lls[-1])

plt.figure(figsize=(5, 5))
plt.plot(np.ones(len(lls)) * true_ll, '-r')
plt.plot(lls)
plt.show()

_, hmm_z = hmm.viterbi(x)
_seq = npr.choice(len(x))
hmm.permute(permutation(true_z[_seq], hmm_z[_seq], K1=3, K2=3))

_, hmm_z = hmm.viterbi(x[_seq])

plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.imshow(true_z[_seq][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
plt.xlim(0, len(x[_seq]))
plt.ylabel("$z_{\\mathrm{true}}$")
plt.yticks([])

plt.subplot(212)
plt.imshow(hmm_z[0][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
plt.xlim(0, len(x[_seq]))
plt.ylabel("$z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")

plt.tight_layout()
plt.show()

hmm_x = hmm.mean_observation(x)

plt.figure(figsize=(8, 4))
plt.plot(x[_seq] + 10 * np.arange(hmm.dm_obs), '-k', lw=2)
plt.plot(hmm_x[_seq] + 10 * np.arange(hmm.dm_obs), '-', lw=2)
plt.show()
