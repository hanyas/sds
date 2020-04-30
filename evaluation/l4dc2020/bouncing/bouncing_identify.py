import numpy as np
import numpy.random as npr

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

    env = gym.make('BouncingBall-ID-v0')
    env._max_episode_steps = 5000
    env.unwrapped._dt = 0.05
    env.unwrapped._sigma = 1e-16
    env.seed(1337)

    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    nb_train_rollouts, nb_train_steps = 25, 150
    nb_test_rollouts, nb_test_steps = 5, 150

    train_obs, train_act = sample_env(env, nb_train_rollouts, nb_train_steps)
    test_obs, test_act = sample_env(env, nb_test_rollouts, nb_test_steps)

    # fig, ax = plt.subplots(nrows=1, ncols=dm_obs + dm_act, figsize=(12, 4))
    # for _obs, _act in zip(obs, act):
    #     for k, col in enumerate(ax[:-1]):
    #         col.plot(_obs[:, k])
    #     ax[-1].plot(_act)
    # plt.show()

    nb_states = 2

    obs_prior = {'mu0': 0., 'sigma0': 1e64, 'nu0': (dm_obs + 1) * 10, 'psi0': 1e-16 * 10}
    obs_mstep_kwargs = {'use_prior': True}

    trans_type = 'neural'
    trans_prior = {'l2_penalty': 1e-32, 'alpha': 1, 'kappa': 50}
    trans_kwargs = {'hidden_layer_sizes': (16,),
                    'norm': {'mean': np.array([0., 0., 0.]),
                             'std': np.array([1., 1., 1.])}}
    trans_mstep_kwargs = {'nb_iter': 25, 'batch_size': 256, 'lr': 1e-3}

    rarhmm = rARHMM(nb_states, dm_obs, dm_act,
                    trans_type=trans_type,
                    obs_prior=obs_prior,
                    trans_prior=trans_prior,
                    trans_kwargs=trans_kwargs)
    rarhmm.initialize(train_obs, train_act)

    lls = rarhmm.em(train_obs, train_act,
                    nb_iter=100, prec=1e-2,
                    obs_mstep_kwargs=obs_mstep_kwargs,
                    trans_mstep_kwargs=trans_mstep_kwargs)

    # plt.figure(figsize=(5, 5))
    # plt.plot(lls)
    # plt.show()
    #
    # plt.figure(figsize=(8, 8))
    # _, state = rarhmm.viterbi(train_obs, train_act)
    # _seq = npr.choice(len(train_obs))
    #
    # plt.subplot(211)
    # plt.plot(train_obs[_seq])
    # plt.xlim(0, len(train_obs[_seq]))
    #
    # plt.subplot(212)
    # plt.imshow(state[_seq][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    # plt.xlim(0, len(train_obs[_seq]))
    # plt.ylabel("$z_{\\mathrm{inferred}}$")
    # plt.yticks([])
    #
    # plt.show()

    # torch.save(rarhmm, open(rarhmm.trans_type + "_rarhmm_bouncing.pkl", "wb"))

    hr = [1, 20, 40, 60, 80, 100]
    for h in hr:
        print("MSE: {0[0]}, EVAR:{0[1]}".format(rarhmm.kstep_mse(test_obs, test_act, horizon=h, mix=False)))
