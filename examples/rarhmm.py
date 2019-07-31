#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: rarhmm.py
# @Date: 2019-07-30-21-01
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np
np.set_printoptions(precision=5, suppress=True)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sds import rARHMM
from sds.utils import permutation

import matplotlib.pyplot as plt
from hips.plotting.colormaps import gradient_cmap

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

color_names = ["windows blue", "red", "amber", "faded green", "dusty purple", "orange"]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

true_rarhmm = rARHMM(nb_states=3, dm_obs=2, dm_act=0)

# trajectory lengths
T = [1250, 1150, 1025]

# empty action sequence
act = [np.zeros((t, 0)) for t in T]

true_z, y = true_rarhmm.sample(T=T, act=act)
true_ll = true_rarhmm.log_probability(y, act)

rarhmm = rARHMM(nb_states=3, dm_obs=2, dm_act=0)
rarhmm.initialize(y, act)

lls = rarhmm.em(y, act, nb_iter=50, prec=1e-4, verbose=True)
print("true_ll=", true_ll, "hmm_ll=", lls[-1])

plt.figure(figsize=(5, 5))
plt.plot(np.ones(len(lls)) * true_ll, '-r')
plt.plot(lls)
plt.show()

_seq = np.random.choice(len(y))
rarhmm.permute(permutation(true_z[_seq], rarhmm.viterbi([y[_seq]], [act[_seq]])[1][0]))
_, rarhmm_z = rarhmm.viterbi([y[_seq]], [act[_seq]])

plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.imshow(true_z[_seq][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
plt.xlim(0, len(y[_seq]))
plt.ylabel("$z_{\\mathrm{true}}$")
plt.yticks([])

plt.subplot(212)
plt.imshow(rarhmm_z[0][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
plt.xlim(0, len(y[_seq]))
plt.ylabel("$z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")

plt.tight_layout()
plt.show()

rarhmm_y = rarhmm.mean_observation(y, act)

plt.figure(figsize=(8, 4))
plt.plot(y[_seq] + 10 * np.arange(rarhmm.dm_obs), '-k', lw=2)
plt.plot(rarhmm_y[_seq] + 10 * np.arange(rarhmm.dm_obs), '-', lw=2)
plt.show()
