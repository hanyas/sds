import autograd.numpy as np
import autograd.numpy.random as npr

from sds import rARHMM
from sds.utils import sample_env

from joblib import Parallel, delayed

import multiprocessing
nb_cores = multiprocessing.cpu_count()


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
    import sds

    env = gym.make('Cartpole-ID-v1')
    env._max_episode_steps = 5000
    env.unwrapped._dt = 0.01
    env.unwrapped._sigma = 1e-8

    nb_rollouts, nb_steps = 25, 250
    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    obs, act = sample_env(env, nb_rollouts, nb_steps)

    nb_states = 7

    obs_prior = {'mu0': 0., 'sigma0': 1e64, 'nu0': (dm_obs + 1) + 10, 'psi0': 1e-8 * 10}
    obs_mstep_kwargs = {'use_prior': True}

    trans_type = 'neural'
    trans_prior = {'l2_penalty': 0., 'alpha': 1, 'kappa': 5}
    trans_kwargs = {'hidden_layer_sizes': (25,),
                    'norm': {'mean': np.array([0., 0., 0., 0., 0., 0.]),
                             'std': np.array([5., 1., 1., 5., 10., 5.])}}
    trans_mstep_kwargs = {'nb_iter': 25, 'batch_size': 128, 'lr': 1e-4}

    models, lls, scores = parallel_em(nb_jobs=6,
                                      nb_states=nb_states, obs=obs, act=act,
                                      trans_type=trans_type,
                                      obs_prior=obs_prior,
                                      trans_prior=trans_prior,
                                      trans_kwargs=trans_kwargs,
                                      obs_mstep_kwargs=obs_mstep_kwargs,
                                      trans_mstep_kwargs=trans_mstep_kwargs,
                                      nb_iter=500, prec=1e-4)
    rarhmm = models[np.argmax(scores)]

    print("rarhmm, stochastic, " + rarhmm.trans_type)
    print(np.c_[lls, scores])

    plt.figure(figsize=(8, 8))
    _, state = rarhmm.viterbi(obs, act)
    _seq = npr.choice(len(obs))

    plt.subplot(211)
    plt.plot(obs[_seq])
    plt.xlim(0, len(obs[_seq]))

    plt.subplot(212)
    plt.imshow(state[_seq][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    plt.xlim(0, len(obs[_seq]))
    plt.ylabel("$z_{\\mathrm{inferred}}$")
    plt.yticks([])

    plt.show()

    # torch.save(rarhmm, open(rarhmm.trans_type + "_rarhmm_cartpole_polar.pkl", "wb"))
    # print(rarhmm.kstep_mse(obs[0:5], act[0:5], horizon=5, mix=False))
