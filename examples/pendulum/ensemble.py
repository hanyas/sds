import numpy as np
import numpy.random as npr

from sds_numpy import Ensemble
from sds_numpy.utils import sample_env


if __name__ == "__main__":

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

    import os
    import random
    import torch

    import gym
    import sds_numpy

    random.seed(1337)
    npr.seed(1337)
    torch.manual_seed(1337)
    torch.set_num_threads(1)

    env = gym.make('Pendulum-ID-v1')
    env._max_episode_steps = 5000
    env.unwrapped._dt = 0.01
    env.unwrapped._sigma = 1e-4
    env.seed(1337)

    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    nb_train_rollouts, nb_train_steps = 15, 250
    nb_test_rollouts, nb_test_steps = 5, 100

    train_obs, train_act = sample_env(env, nb_train_rollouts, nb_train_steps)
    test_obs, test_act = sample_env(env, nb_test_rollouts, nb_test_steps)

    nb_states = 7

    obs_prior = {'mu0': 0., 'sigma0': 1e64,
                 'nu0': (dm_obs + 1) + 23, 'psi0': 1e-4 * 23}
    obs_mstep_kwargs = {'use_prior': True}

    trans_type = 'neural'
    trans_prior = {'l2_penalty': 1e-32, 'alpha': 1, 'kappa': 1}
    trans_kwargs = {'hidden_layer_sizes': (24,),
                    'norm': {'mean': np.array([0., 0., 0., 0.]),
                             'std': np.array([1., 1., 8., 2.5])}}
    trans_mstep_kwargs = {'nb_iter': 50, 'batch_size': 256, 'lr': 5e-4}

    ensemble = Ensemble(nb_states=nb_states, type='rarhmm',
                        dm_obs=dm_obs, dm_act=dm_act,
                        trans_type=trans_type,
                        obs_prior=obs_prior,
                        trans_prior=trans_prior,
                        trans_kwargs=trans_kwargs)

    lls = ensemble.em(train_obs, train_act,
                      nb_iter=200, prec=1e-2,
                      obs_mstep_kwargs=obs_mstep_kwargs,
                      trans_mstep_kwargs=trans_mstep_kwargs)

    print([_ll[-1] for _ll in lls])

    hr = [1, 5, 10, 15, 20, 25]
    for h in hr:
        _mse, _smse, _evar = ensemble.kstep_mse(test_obs, test_act, horizon=h)
        print(f"MSE: {_mse}, SMSE:{_smse}, EVAR:{_evar}")
