import autograd.numpy as np

from sds import erARHMM
from sds.utils import sample_env

import random
from joblib import Parallel, delayed


def create_job(kwargs):
    # model arguments
    nb_states = kwargs.pop('nb_states')
    trans_type = kwargs.pop('trans_type')
    obs_prior = kwargs.pop('obs_prior')
    ctl_prior = kwargs.pop('ctl_prior')
    trans_prior = kwargs.pop('trans_prior')
    trans_kwargs = kwargs.pop('trans_kwargs')

    # em arguments
    obs = kwargs.pop('obs')
    act = kwargs.pop('act')
    prec = kwargs.pop('prec')
    nb_iter = kwargs.pop('nb_iter')
    obs_mstep_kwargs = kwargs.pop('obs_mstep_kwargs')
    ctl_mstep_kwargs = kwargs.pop('ctl_mstep_kwargs')
    trans_mstep_kwargs = kwargs.pop('trans_mstep_kwargs')

    train_obs, train_act = zip(*random.sample(list(zip(obs, act)), int(0.8 * len(obs))))

    dm_obs = train_obs[0].shape[-1]
    dm_act = train_act[0].shape[-1]

    erarhmm = erARHMM(nb_states, dm_obs, dm_act,
                      trans_type=trans_type,
                      obs_prior=obs_prior,
                      ctl_prior=ctl_prior,
                      trans_prior=trans_prior,
                      trans_kwargs=trans_kwargs,
                      learn_ctl=False)
    # erarhmm.initialize(train_obs, train_act)

    erarhmm.em(train_obs, train_act,
               nb_iter=nb_iter, prec=prec, verbose=False,
               obs_mstep_kwargs=obs_mstep_kwargs,
               ctl_mstep_kwargs=ctl_mstep_kwargs,
               trans_mstep_kwargs=trans_mstep_kwargs)

    nb_train = np.vstack(train_obs).shape[0]
    nb_all = np.vstack(obs).shape[0]

    train_ll = erarhmm.log_norm(train_obs, train_act)
    all_ll = erarhmm.log_norm(obs, act)

    score = (all_ll - train_ll) / (nb_all - nb_train)

    return erarhmm, all_ll, score


def parallel_em(nb_jobs=50, **kwargs):
    kwargs_list = [kwargs for _ in range(nb_jobs)]
    results = Parallel(n_jobs=-1, verbose=10, backend='loky', )(map(delayed(create_job), kwargs_list))
    erarhmms, lls, scores = list(map(list, zip(*results)))
    return erarhmms, lls, scores


if __name__ == "__main__":

    import os
    import pickle

    import gym
    import rl

    env = gym.make('Pendulum-RL-v0')
    env._max_episode_steps = 5000

    nb_rollouts, nb_steps = 50, 100
    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    obs, act = sample_env(env, nb_rollouts, nb_steps)

    nb_states = 5

    obs_prior = {'mu0': 0., 'sigma0': 1e16, 'nu0': dm_obs + 2, 'psi0': 1e-2}
    ctl_prior = {'mu0': 0., 'sigma0': 1e16, 'nu0': dm_act + 2, 'psi0': 5e-1}
    trans_prior = {'l2': 1e-16, 'alpha': 1, 'kappa': 100}

    obs_mstep_kwargs = {'use_prior': False}
    ctl_mstep_kwargs = {'use_prior': False}

    trans_type = 'poly'
    trans_kwargs = {'degree': 1}
    trans_mstep_kwargs = {'nb_iter': 25, 'batch_size': 512, 'lr': 1e-3}

    # trans_type = 'neural'
    # trans_kwargs = {'hidden_layer_sizes': (10,)}
    # trans_mstep_kwargs = {'nb_iter': 25, 'batch_size': None, 'lr': 1e-3}

    models, lls, scores = parallel_em(nb_jobs=12,
                                      nb_states=nb_states, obs=obs, act=act,
                                      trans_type=trans_type,
                                      obs_prior=obs_prior,
                                      ctl_prior=ctl_prior,
                                      trans_prior=trans_prior,
                                      trans_kwargs=trans_kwargs,
                                      obs_mstep_kwargs=obs_mstep_kwargs,
                                      ctl_mstep_kwargs=ctl_mstep_kwargs,
                                      trans_mstep_kwargs=trans_mstep_kwargs,
                                      nb_iter=100, prec=1e-4)
    erarhmm = models[np.argmax(scores)]

    print("erarhmm, stochastic, " + erarhmm.trans_type)
    print(np.c_[lls, scores])

    # pickle.dump(erarhmm, open(erarhmm.trans_type + "_erarhmm_pendulum_polar.pkl", "wb"))
    # pickle.dump(erarhmm, open(erarhmm.trans_type + "_erarhmm_pendulum_cart.pkl", "wb"))
