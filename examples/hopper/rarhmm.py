import autograd.numpy as np
import autograd.numpy.random as npr

import torch

from sds import rARHMM
from sds.utils import sample_env

import random


if __name__ == "__main__":

    random.seed(1337)
    npr.seed(1337)
    torch.manual_seed(1337)
    torch.set_num_threads(1)

    import matplotlib.pyplot as plt

    from hips.plotting.colormaps import gradient_cmap
    import seaborn as sns

    sns.set_style("white")
    sns.set_context("talk")

    color_names = ["windows blue", "red", "amber",
                   "faded green", "dusty purple",
                   "orange", "clay", "pink", "greyish",
                   "mint", "light cyan", "steel blue",
                   "forest green", "pastel purple",
                   "salmon", "dark brown"]

    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors)

    import torch
    import gym
    import sds

    import pickle
    data = pickle.load(open("sac_hopper.pkl", "rb"))
    raw_obs, raw_act = data['obs'], data['act']

    nb_rollouts, nb_steps = 50, 300
    dm_obs, dm_act = 6, 3

    from scipy import signal
    fs = 240
    fc = 30  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = signal.butter(2, w, 'lowpass', output='ba')

    obs, act = [], []
    for _obs, _act in zip(raw_obs[:nb_rollouts], raw_act[:nb_rollouts]):
        _in = _obs[50:nb_steps, 8:14]
        _out = signal.filtfilt(b, a, _in.T).T
        obs.append(_out)

        _in = _act[50:nb_steps, :]
        _out = signal.filtfilt(b, a, _in.T).T
        act.append(_out)

    # fig, ax = plt.subplots(nrows=1, ncols=dm_obs + dm_act, figsize=(12, 4))
    # for _obs, _act in zip(obs[:1], act[:1]):
    #     for k, col in enumerate(ax[:-dm_act]):
    #         col.plot(_obs[:, k])
    #     for k, col in enumerate(ax[dm_obs:]):
    #         col.plot(_act[:, k])
    # plt.show()

    nb_states = 13

    obs_prior = {'mu0': 0., 'sigma0': 1e64, 'nu0': (dm_obs + 1) * 10, 'psi0': 1e-16 * 10}
    obs_mstep_kwargs = {'use_prior': True}

    trans_type = 'neural'
    trans_prior = {'l2_penalty': 1e-32, 'alpha': 1, 'kappa': 2500}
    trans_kwargs = {'hidden_layer_sizes': (64,),
                    'norm': {'mean': np.zeros((dm_obs + dm_act, )),
                             'std': np.ones((dm_obs + dm_act, ))}}
    trans_mstep_kwargs = {'nb_iter': 50, 'batch_size': 512, 'lr': 1e-3}

    rarhmm = rARHMM(nb_states, dm_obs, dm_act,
                    trans_type=trans_type,
                    obs_prior=obs_prior,
                    trans_prior=trans_prior,
                    trans_kwargs=trans_kwargs)
    rarhmm.initialize(obs, act)

    lls = rarhmm.em(obs, act, nb_iter=50, prec=1e-4,
                    obs_mstep_kwargs=obs_mstep_kwargs,
                    trans_mstep_kwargs=trans_mstep_kwargs)

    # plt.figure(figsize=(5, 5))
    # plt.plot(lls)
    # plt.show()
    #
    # plt.figure(figsize=(8, 8))
    # idx = npr.choice(nb_rollouts)
    # _, state = rarhmm.viterbi(obs, act)
    # _seq = npr.choice(len(obs))
    #
    # plt.subplot(211)
    # plt.plot(obs[_seq][:250])
    # plt.xlim(0, len(obs[_seq][:250]))
    #
    # plt.subplot(212)
    # plt.imshow(state[_seq][None, :250], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    # plt.xlim(0, len(obs[_seq][:250]))
    # plt.ylabel("$z_{\\mathrm{inferred}}$")
    # plt.yticks([])
    #
    # plt.show()

    hr = [1, 3, 6, 9, 12, 15]
    for h in hr:
        print("MSE: {0[0]}, EVAR:{0[1]}".format(rarhmm.kstep_mse(obs[0:5], act[0:5], horizon=h, mix=False)))
