import autograd.numpy as np
import autograd.numpy.random as npr

from sds import rARHMM
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

    rarhmm = rARHMM(nb_states, dm_obs, dm_act, type)
    rarhmm.initialize(_train_obs, _train_act)
    lls = rarhmm.em(_train_obs, _train_act, nb_iter=nb_iter, prec=prec, verbose=False)

    _range = np.arange(nb_rollouts)
    _mask = np.ones(nb_rollouts, dtype=bool)
    _mask[_train_choice] = False
    _test_choice = _range[_mask]

    _test_obs = [obs[i] for i in _test_choice]
    _test_act = [act[i] for i in _test_choice]
    mse, norm_mse = rarhmm.kstep_mse(obs, act, horizon=25, stoch=False)

    return rarhmm, lls, mse, norm_mse


def parallel_em(obs, act, nb_states, type, nb_iter=50, prec=1e-4, nb_jobs=50):
    args = [(obs, act, nb_states, type, nb_iter, prec) for _ in range(nb_jobs)]
    results = Parallel(n_jobs=nb_jobs, verbose=10, backend='loky')(map(delayed(create_job), args))
    rarhmms, lls, mse, norm_mse = list(map(list, zip(*results)))
    return rarhmms, lls, mse, norm_mse


if __name__ == "__main__":

    import os
    import pickle

    import gym
    import rl

    env = gym.make('Pendulum-RL-v0')
    env._max_episode_steps = 5000

    nb_rollouts, nb_steps = 50, 200
    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    obs, act = sample_env(env, nb_rollouts, nb_steps)

    nb_states = 5
    models, liklhds, mse, norm_mse = parallel_em(obs=obs, act=act,
                                                 nb_states=nb_states,
                                                 type='neural-recurrent',
                                                 nb_iter=100,
                                                 prec=1e-4, nb_jobs=100)

    all_models = []
    for _nmse in norm_mse:
        all_models.append(_nmse)
    rarhmm = models[np.argmax(all_models)]

    print("rarhmm, stochastic, neural")
    print(np.c_[mse, norm_mse])

    pickle.dump(rarhmm, open("rarhmm_pendulum_neural.pkl", "wb"))
