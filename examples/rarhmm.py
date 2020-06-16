import numpy as np
import numpy.random as npr

from sds_numpy import rARHMM
from sds_numpy.utils import permutation

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

true_rarhmm = rARHMM(nb_states=3, dm_obs=2, trans_type='poly')

# trajectory lengths
T = [1250, 1150, 1025]

true_z, x = true_rarhmm.sample(horizon=T)
true_ll = true_rarhmm.log_norm(x)

rarhmm = rARHMM(nb_states=3, dm_obs=2,
                trans_type='poly', preprocess=True)
rarhmm.initialize(x)

lls = rarhmm.em(x, nb_iter=100, prec=0.)
print("true_ll=", true_ll, "hmm_ll=", lls[-1])

plt.figure(figsize=(5, 5))
plt.plot(np.ones(len(lls)) * true_ll, '-r')
plt.plot(lls)
plt.show()

_, rarhmm_z = rarhmm.viterbi(x)
_seq = npr.choice(len(x))
rarhmm.permute(permutation(true_z[_seq], rarhmm_z[_seq], K1=3, K2=3))

_, rarhmm_z = rarhmm.viterbi(x[_seq])

plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.imshow(true_z[_seq][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
plt.xlim(0, len(x[_seq]))
plt.ylabel("$z_{\\mathrm{true}}$")
plt.yticks([])

plt.subplot(212)
plt.imshow(rarhmm_z[0][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
plt.xlim(0, len(x[_seq]))
plt.ylabel("$z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")

plt.tight_layout()
plt.show()

rarhmm_x = rarhmm.mean_observation(x)

plt.figure(figsize=(8, 4))
plt.plot(x[_seq] + 10 * np.arange(rarhmm.dm_obs), '-k', lw=2)
plt.plot(rarhmm_x[_seq] + 10 * np.arange(rarhmm.dm_obs), '-', lw=2)
plt.show()
