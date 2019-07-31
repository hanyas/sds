#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: hmm.py
# @Date: 2019-07-30-20-56
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np
np.set_printoptions(precision=5, suppress=True)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

true_hmm = HMM(nb_states=3, dm_obs=2, dm_act=0)

thetas = np.linspace(0, 2 * np.pi, true_hmm.nb_states, endpoint=False)
for k in range(true_hmm.nb_states):
    true_hmm.observations.mu[k, :] = 3 * np.array([np.cos(thetas[k]), np.sin(thetas[k])])

# trajectory lengths
T = [95, 85, 75]

# empty action sequence
act = [np.zeros((t, 0)) for t in T]

true_z, y = true_hmm.sample(T=T, act=act)
true_ll = true_hmm.log_probability(y, act=act)

hmm = HMM(nb_states=3, dm_obs=2, dm_act=0)
hmm.initialize(y, act=None)

lls = hmm.em(y, act, nb_iter=50, prec=1e-24, verbose=True)
print("true_ll=", true_ll, "hmm_ll=", lls[-1])

plt.figure(figsize=(5, 5))
plt.plot(np.ones(len(lls)) * true_ll, '-r')
plt.plot(lls)
plt.show()

_seq = np.random.choice(len(y))
hmm.permute(permutation(true_z[_seq], hmm.viterbi([y[_seq]], [act[_seq]])[1][0]))
_, hmm_z = hmm.viterbi([y[_seq]], [act[_seq]])

plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.imshow(true_z[_seq][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
plt.xlim(0, len(y[_seq]))
plt.ylabel("$z_{\\mathrm{true}}$")
plt.yticks([])

plt.subplot(212)
plt.imshow(hmm_z[0][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
plt.xlim(0, len(y[_seq]))
plt.ylabel("$z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")

plt.tight_layout()
plt.show()

hmm_y = hmm.mean_observation(y, act)

plt.figure(figsize=(8, 4))
plt.plot(y[_seq] + 10 * np.arange(hmm.dm_obs), '-k', lw=2)
plt.plot(hmm_y[_seq] + 10 * np.arange(hmm.dm_obs), '-', lw=2)
plt.show()
