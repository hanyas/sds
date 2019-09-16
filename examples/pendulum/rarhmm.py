import autograd.numpy as np
import autograd.numpy.random as npr

import torch

from sds import rARHMM
from sds.utils import sample_env


if __name__ == "__main__":

    np.random.seed(1337)
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

    import pickle
    import gym
    import rl

    env = gym.make('Pendulum-RL-v0')
    env._max_episode_steps = 5000
    env.seed(1337)

    nb_rollouts, nb_steps = 25, 200
    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    obs, act = sample_env(env, nb_rollouts, nb_steps)

    nb_states = 5
    obs_prior = {'mu0': 0., 'sigma0': 1.e12, 'nu0': dm_obs + 2, 'psi0': 1.e-4}
    # trans_kwargs = {'hidden_layer_sizes': (10,)}
    trans_kwargs = {'degree': 3}
    rarhmm = rARHMM(nb_states, dm_obs, dm_act, trans_type='poly',
                    obs_prior=obs_prior, trans_kwargs=trans_kwargs)
    rarhmm.initialize(obs, act)

    lls = rarhmm.em(obs, act, nb_iter=50, prec=0., verbose=True)

    plt.figure(figsize=(5, 5))
    plt.plot(lls)
    plt.show()

    plt.figure(figsize=(8, 8))
    idx = npr.choice(nb_rollouts)
    _, sample_obs = rarhmm.sample([act[idx]], horizon=[nb_steps])
    plt.plot(sample_obs[0])
    plt.show()
