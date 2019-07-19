#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: rarahmm_test.py
# @Date: 2019-06-05-15-27
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import numpy as np
from sds import rARHMM


# list of dicts to dict of lists
def lod2dol(*dicts):
    d = {}
    for dict in dicts:
        for key in dict:
            try:
                d[key].append(dict[key])
            except KeyError:
                d[key] = [dict[key]]

    return d


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from hips.plotting.colormaps import gradient_cmap
    import seaborn as sns

    sns.set_style("white")
    sns.set_context("talk")

    color_names = [
        "windows blue",
        "red",
        "amber",
        "faded green",
        "dusty purple",
        "orange"
    ]

    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors)

    file = np.load('reps_pendulum_rollouts.npz', allow_pickle=True)
    rollouts = file['arr_0']
    data = lod2dol(*rollouts)

    rarhmm = rARHMM(nb_states=3, dm_obs=3, dm_act=1, type='recurrent')
    rarhmm.initialize(data['x'], data['u'])
    lls = rarhmm.em(data['x'],  data['u'], nb_iter=50, prec=1e-4, verbose=True)

    plt.figure(figsize=(5, 5))
    plt.plot(lls)
    plt.show()

    _seq = np.random.choice(len(data['x']))
    _, z = rarhmm.viterbi([data['x'][_seq]], [data['u'][_seq]])

    x = rarhmm.mean_observation(data['x'], data['u'])

    plt.figure(figsize=(8, 8))
    plt.subplot(211)
    plt.imshow(z[0][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    plt.xlim(0, len(z[0]))
    plt.ylabel("$state_{\\mathrm{true}}$")
    plt.yticks([])

    plt.subplot(212)
    plt.plot(x[_seq], '-k', lw=2)
    plt.xlim(0, len(x[_seq]))
    plt.ylabel("$obs_{\\mathrm{inferred}}$")
    plt.xlabel("time")

    plt.tight_layout()
    plt.show()
