import autograd.numpy as np
import autograd.numpy.random as npr

from sds import erARHMM
from sds.utils import sample_env

from joblib import Parallel, delayed


def create_job(args):
    obs, act, nb_states, type, nb_iter, prec = args

    dm_obs = obs[0].shape[-1]
    dm_act = act[0].shape[-1]

    nb_rollouts = len(obs)
    _train_choice = npr.choice(nb_rollouts, size=int(0.8 * nb_rollouts), replace=False)
    _train_obs = [obs[i] for i in _train_choice]
    _train_act = [act[i] for i in _train_choice]

    erarhmm = erARHMM(nb_states, dm_obs, dm_act, type, learn_ctl=False)
    erarhmm.initialize(_train_obs, _train_act)
    lls = erarhmm.em(_train_obs, _train_act, nb_iter=nb_iter, prec=prec, verbose=False)

    mse, norm_mse = erarhmm.kstep_mse(obs, act, horizon=25, stoch=False)
    return erarhmm, lls[-1], mse, norm_mse


def parallel_em(obs, act, nb_states, type, nb_iter=50, prec=1e-4, nb_jobs=50):
    args = [(obs, act, nb_states, type, nb_iter, prec) for _ in range(nb_jobs)]
    results = Parallel(n_jobs=-1, verbose=10, backend='loky',
                       max_nbytes='1000M')(map(delayed(create_job), args))
    erarhmms, lls, mse, norm_mse = list(map(list, zip(*results)))
    return erarhmms, lls, mse, norm_mse


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
    models, liklhds, mse, norm_mse = parallel_em(obs=obs, act=act,
                                                 nb_states=nb_states,
                                                 type='neural-recurrent',
                                                 nb_iter=150,
                                                 prec=1e-4, nb_jobs=32)
    erarhmm = models[np.argmax(norm_mse)]

    print("erarhmm, stochastic, neural")
    print(np.c_[liklhds, mse, norm_mse])

    pickle.dump(erarhmm, open("neural_erarhmm_pendulum_polar.pkl", "wb"))
    # pickle.dump(erarhmm, open("neural_erarhmm_pendulum_cart.pkl", "wb"))
