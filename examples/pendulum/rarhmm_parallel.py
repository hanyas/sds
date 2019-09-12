import autograd.numpy as np

from sds import rARHMM
from sds.utils import sample_env

import random
from joblib import Parallel, delayed


def create_job(kwargs):
    obs = kwargs.pop('obs')
    act = kwargs.pop('act')
    nb_states = kwargs.pop('nb_states')

    type = kwargs.pop('type')
    prec = kwargs.pop('prec')
    nb_iter = kwargs.pop('nb_iter')

    train_obs, train_act = zip(*random.sample(list(zip(obs, act)), int(0.8 * len(obs))))

    dm_obs = train_obs[0].shape[-1]
    dm_act = train_act[0].shape[-1]

    rarhmm = rARHMM(nb_states, dm_obs, dm_act, type)
    rarhmm.initialize(train_obs, train_act)
    lls = rarhmm.em(train_obs, train_act, nb_iter=nb_iter, prec=prec, verbose=False)

    mse, norm_mse = rarhmm.kstep_mse(obs, act, horizon=10, stoch=False)
    return rarhmm, lls[-1], mse, norm_mse


def parallel_em(nb_jobs=50, **kwargs):
    kwargs_list = [kwargs for _ in range(nb_jobs)]
    results = Parallel(n_jobs=-1, verbose=10, backend='loky',
                       max_nbytes='1000M')(map(delayed(create_job), kwargs_list))
    rarhmms, lls, mse, norm_mse = list(map(list, zip(*results)))
    return rarhmms, lls, mse, norm_mse


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
    models, liklhds, mse, norm_mse = parallel_em(nb_jobs=32,
                                                 obs=obs, act=act,
                                                 nb_states=nb_states,
                                                 type='neural-recurrent',
                                                 nb_iter=150, prec=1e-4)
    rarhmm = models[np.argmax(norm_mse)]

    print("rarhmm, stochastic, neural")
    print(np.c_[liklhds, mse, norm_mse])

    pickle.dump(rarhmm, open("neural_rarhmm_pendulum_polar.pkl", "wb"))
    # pickle.dump(rarhmm, open("neural_rarhmm_pendulum_cart.pkl", "wb"))
