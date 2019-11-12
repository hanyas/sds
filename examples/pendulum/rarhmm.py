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

    import matplotlib.pyplot as plt

    from hips.plotting.colormaps import gradient_cmap
    import seaborn as sns

    sns.set_style("white")
    sns.set_context("talk")

    color_names = ["windows blue", "red", "amber",
                   "faded green", "dusty purple", "orange"]

    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors)

    import torch
    import gym
    import rl

    env = gym.make('Pendulum-RL-v0')
    env._max_episode_steps = 5000
    env.unwrapped._dt = 0.01
    env.unwrapped._sigma = 1e-8
    env.seed(1337)

    nb_rollouts, nb_steps = 50, 250
    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    obs, act = sample_env(env, nb_rollouts, nb_steps)

    # fig, ax = plt.subplots(nrows=1, ncols=dm_obs + dm_act, figsize=(12, 4))
    # for _obs, _act in zip(obs, act):
    #     for k, col in enumerate(ax[:-1]):
    #         col.plot(_obs[:, k])
    #     ax[-1].plot(_act)
    # # plt.show()

    nb_states = 5

    obs_prior = {'mu0': 0., 'sigma0': 1e16, 'nu0': dm_obs + 2, 'psi0': 1.0}
    trans_prior = {'l2': 1e-16, 'alpha': 1, 'kappa': 100}

    obs_mstep_kwargs = {'use_prior': False}

    trans_type = 'neural'
    trans_kwargs = {'hidden_layer_sizes': (10,),
                    'norm': {'mean': np.array([0., 0., 0.]),
                             'std': np.array([np.pi, 8., 2.5])}}
    trans_mstep_kwargs = {'nb_iter': 100}  # 'batch_size': 1024, 'lr': 1e-3}

    # trans_type = 'poly'
    # trans_kwargs = {'degree': 1,
    #                 'norm': {'mean': np.array([0., 0., 0.]),
    #                          'std': np.array([np.pi, 8., 2.5])}}
    # trans_mstep_kwargs = {'nb_iter': 100, 'batch_size': 1024, 'lr': 1e-3}

    rarhmm = rARHMM(nb_states, dm_obs, dm_act,
                    trans_type=trans_type,
                    obs_prior=obs_prior,
                    trans_prior=trans_prior,
                    trans_kwargs=trans_kwargs)
    # rarhmm.initialize(obs, act)

    rarhmm.stochastic_em(obs, act, nb_epochs=250, verbose=True,
                         method='adam', step_size=5e-4)

    lls = rarhmm.em(obs, act, nb_iter=10, prec=0., verbose=True,
                    obs_mstep_kwargs=obs_mstep_kwargs,
                    trans_mstep_kwargs=trans_mstep_kwargs)

    plt.figure(figsize=(5, 5))
    plt.plot(lls)
    plt.show()

    plt.figure(figsize=(8, 8))
    idx = npr.choice(nb_rollouts)
    _, sample_obs = rarhmm.sample([act[idx]], horizon=[nb_steps])
    plt.plot(sample_obs[0])
    plt.show()

    # torch.save(rarhmm, open(rarhmm.trans_type + "_rarhmm_pendulum_polar.pkl", "wb"))
    # torch.save(rarhmm, open(rarhmm.trans_type + "_rarhmm_pendulum_cart.pkl", "wb"))
