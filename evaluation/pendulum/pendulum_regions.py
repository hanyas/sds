import autograd.numpy as np
import autograd.numpy.random as npr

from sds import rARHMM
from sds.utils import sample_env

from joblib import Parallel, delayed

import multiprocessing
nb_cores = multiprocessing.cpu_count()


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


def create_job(kwargs):
    # model arguments
    nb_states = kwargs.pop('nb_states')
    trans_type = kwargs.pop('trans_type')
    obs_prior = kwargs.pop('obs_prior')
    trans_prior = kwargs.pop('trans_prior')
    trans_kwargs = kwargs.pop('trans_kwargs')

    # em arguments
    obs = kwargs.pop('obs')
    act = kwargs.pop('act')
    prec = kwargs.pop('prec')
    nb_iter = kwargs.pop('nb_iter')
    obs_mstep_kwargs = kwargs.pop('obs_mstep_kwargs')
    trans_mstep_kwargs = kwargs.pop('trans_mstep_kwargs')

    train_obs, train_act, test_obs, test_act = [], [], [], []
    train_idx = npr.choice(a=len(obs), size=int(0.8 * len(obs)), replace=False)
    for i in range(len(obs)):
        if i in train_idx:
            train_obs.append(obs[i])
            train_act.append(act[i])
        else:
            test_obs.append(obs[i])
            test_act.append(act[i])

    dm_obs = train_obs[0].shape[-1]
    dm_act = train_act[0].shape[-1]

    rarhmm = rARHMM(nb_states, dm_obs, dm_act,
                    trans_type=trans_type,
                    obs_prior=obs_prior,
                    trans_prior=trans_prior,
                    trans_kwargs=trans_kwargs)
    # rarhmm.initialize(train_obs, train_act)

    rarhmm.em(train_obs, train_act,
              nb_iter=nb_iter, prec=prec, verbose=True,
              obs_mstep_kwargs=obs_mstep_kwargs,
              trans_mstep_kwargs=trans_mstep_kwargs)

    # rarhmm.earlystop_em(train_obs, train_act,
    #                     nb_iter=nb_iter, prec=prec, verbose=True,
    #                     obs_mstep_kwargs=obs_mstep_kwargs,
    #                     trans_mstep_kwargs=trans_mstep_kwargs,
    #                     test_obs=test_obs, test_act=test_act)

    nb_train = np.vstack(train_obs).shape[0]
    nb_all = np.vstack(obs).shape[0]

    train_ll = rarhmm.log_norm(train_obs, train_act)
    all_ll = rarhmm.log_norm(obs, act)

    score = (all_ll - train_ll) / (nb_all - nb_train)

    return rarhmm, all_ll, score


def parallel_em(nb_jobs=50, **kwargs):
    kwargs_list = [kwargs for _ in range(nb_jobs)]
    results = Parallel(n_jobs=min(nb_jobs, nb_cores), verbose=10, backend='loky')(map(delayed(create_job), kwargs_list))
    rarhmms, lls, scores = list(map(list, zip(*results)))
    return rarhmms, lls, scores


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from hips.plotting.colormaps import gradient_cmap
    import seaborn as sns

    sns.set_style("white")
    sns.set_context("talk")

    color_names = ["windows blue", "red", "amber",
                   "faded green", "dusty purple", "orange"]

    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors)

    import os
    import torch

    import gym
    import rl

    env = gym.make('Pendulum-ID-v0')
    env._max_episode_steps = 5000
    env.unwrapped._dt = 0.01
    env.unwrapped._sigma = 1e-8

    nb_rollouts, nb_steps = 25, 250
    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    train_obs, train_act = sample_env(env, nb_rollouts, nb_steps)
    test_obs, test_act = sample_env(env, 5, nb_steps)

    obs_prior = {'mu0': 0., 'sigma0': 1e32, 'nu0': dm_obs + 2, 'psi0': 1e-8}
    obs_mstep_kwargs = {'use_prior': True}

    trans_type = 'neural'
    trans_prior = {'l2_penalty': 0., 'alpha': 1, 'kappa': 5}
    trans_kwargs = {'hidden_layer_sizes': (25,),
                    'norm': {'mean': np.array([0., 0., 0.]),
                             'std': np.array([np.pi, 8., 2.5])}}
    trans_mstep_kwargs = {'nb_iter': 10, 'batch_size': 512, 'lr': 1e-3}

    nb_states = [3, 4, 5]

    stats = []
    for _nb in nb_states:
        models, lls, scores = parallel_em(nb_jobs=25,
                                          nb_states=_nb, obs=train_obs, act=train_act,
                                          trans_type=trans_type,
                                          obs_prior=obs_prior,
                                          trans_prior=trans_prior,
                                          trans_kwargs=trans_kwargs,
                                          obs_mstep_kwargs=obs_mstep_kwargs,
                                          trans_mstep_kwargs=trans_mstep_kwargs,
                                          nb_iter=200, prec=1e-2)
        _stats = np.zeros((25, 2))
        for m, rarhmm in enumerate(models):
            _stats[m, 0], _stats[m, 1] = rarhmm.kstep_mse(test_obs, test_act, horizon=1, mix=False)

        stats.append(_stats)

    from tikzplotlib import save

    mse = ([_s[:, 0] for _s in stats])
    parts = plt.violinplot(mse, showmeans=False, showmedians=False, showextrema=False)

    for pc in parts['bodies']:
        # pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(mse, [25, 50, 75], axis=1)
    whiskers = np.array([adjacent_values(sorted_array, q1, q3)
                         for sorted_array, q1, q3 in zip(mse, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    plt.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    plt.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

    # set style for the axes
    labels = ['3', '4', '5']
    plt.xticks([1, 2, 3], labels)

    save("rarhmm_pendulum_regions_mse.tex")
    plt.close()

    ##
    evar = ([_s[:, 1] for _s in stats])
    parts = plt.violinplot(evar, showmeans=False, showmedians=False, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(evar, [25, 50, 75], axis=1)
    whiskers = np.array([adjacent_values(sorted_array, q1, q3)
                         for sorted_array, q1, q3 in zip(evar, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    plt.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    plt.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

    # set style for the axes
    labels = ['3', '4', '5']
    plt.xticks([1, 2, 3], labels)

    save("rarhmm_pendulum_regions_evar.tex")
    plt.close()
