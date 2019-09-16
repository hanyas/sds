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

    # em arguments
    obs = kwargs.pop('obs')
    act = kwargs.pop('act')
    prec = kwargs.pop('prec')
    nb_iter = kwargs.pop('nb_iter')

    train_obs, train_act = zip(*random.sample(list(zip(obs, act)), int(0.8 * len(obs))))

    dm_obs = train_obs[0].shape[-1]
    dm_act = train_act[0].shape[-1]

    erarhmm = erARHMM(nb_states, dm_obs, dm_act, trans_type=trans_type,
                    obs_prior=obs_prior, trans_kwargs=trans_kwargs, learn_ctl=False)
    erarhmm.initialize(train_obs, train_act)

    lls = erarhmm.em(train_obs, train_act, nb_iter=nb_iter, prec=prec, verbose=False)
    return erarhmm, lls[-1]


def parallel_em(nb_jobs=50, **kwargs):
    kwargs_list = [kwargs for _ in range(nb_jobs)]
    results = Parallel(n_jobs=-1, verbose=10, backend='loky',
                       max_nbytes='1000M')(map(delayed(create_job), kwargs_list))
    erarhmms, lls = list(map(list, zip(*results)))
    return erarhmms, lls


if __name__ == "__main__":

    import os
    import pickle

    import gym
    import rl

    env = gym.make('Pendulum-RL-v0')
    # env = gym.make('Pendulum-RL-v1')
    env._max_episode_steps = 5000

    nb_rollouts, nb_steps = 50, 200
    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    obs, act = sample_env(env, nb_rollouts, nb_steps)

    nb_states = 5
    obs_prior = {'mu0': 0., 'sigma0': 1.e12, 'nu0': dm_obs + 2, 'psi0': 1.e-4}
    trans_kwargs = {'hidden_layer_sizes': (10,)}
    # trans_kwargs = {'degree': 3}
    models, liklhds = parallel_em(nb_jobs=32,
                                  nb_states=nb_states, obs=obs, act=act,
                                  trans_type='neural',
                                  obs_prior=obs_prior,
                                  trans_kwargs=trans_kwargs,
                                  nb_iter=150, prec=1e-4)
    erarhmm = models[np.argmax(liklhds)]

    print("rarhmm, stochastic, " + erarhmm.trans_type)
    print(np.c_[liklhds])

    # pickle.dump(erarhmm, open(erarhmm.trans_type + "_erarhmm_pendulum_polar.pkl", "wb"))
    # pickle.dump(erarhmm, open(erarhmm.trans_type + "_erarhmm_pendulum_cart.pkl", "wb"))
