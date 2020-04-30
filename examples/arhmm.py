import numpy as np
import numpy.random as npr

from sds import ARHMM
from sds.utils import permutation

import matplotlib.pyplot as plt
from hips.plotting.colormaps import gradient_cmap

import seaborn as sns

import torch

npr.randn(1337)
torch.manual_seed(1337)

sns.set_style("white")
sns.set_context("talk")

color_names = ["windows blue", "red", "amber",
               "faded green", "dusty purple", "orange"]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

true_arhmm = ARHMM(nb_states=3, dm_obs=2)

# trajectory lengths
T = [1250, 1150, 1025]

true_z, x = true_arhmm.sample(horizon=T)
true_ll = true_arhmm.log_norm(x)

arhmm = ARHMM(nb_states=3, dm_obs=2, preprocess=False)
arhmm.initialize(x)

lls = arhmm.em(x, nb_iter=100, prec=0.)
print("true_ll=", true_ll, "hmm_ll=", lls[-1])

plt.figure(figsize=(5, 5))
plt.plot(np.ones(len(lls)) * true_ll, '-r')
plt.plot(lls)
plt.show()

_, arhmm_z = arhmm.viterbi(x)
_seq = npr.choice(len(x))
arhmm.permute(permutation(true_z[_seq], arhmm_z[_seq], K1=3, K2=3))

_, arhmm_z = arhmm.viterbi(x[_seq])

plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.imshow(true_z[_seq][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
plt.xlim(0, len(x[_seq]))
plt.ylabel("$z_{\\mathrm{true}}$")
plt.yticks([])

plt.subplot(212)
plt.imshow(arhmm_z[0][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
plt.xlim(0, len(x[_seq]))
plt.ylabel("$z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")

plt.tight_layout()
plt.show()

arhmm_x = arhmm.mean_observation(x)

plt.figure(figsize=(8, 4))
plt.plot(x[_seq] + 10 * np.arange(arhmm.dm_obs), '-k', lw=2)
plt.plot(arhmm_x[_seq] + 10 * np.arange(arhmm.dm_obs), '-', lw=2)
plt.show()
