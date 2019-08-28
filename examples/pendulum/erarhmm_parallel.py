import autograd.numpy as np
import autograd.numpy.random as npr

from sds import erARHMM
from sds.utils import sample_env

from joblib import Parallel, delayed


def create_job(args):
    nb_states, dm_obs, dm_act, type, obs, act, nb_iter, prec = args

    nb_rollouts = len(obs)
    _choice = npr.choice(nb_rollouts, size=int(0.8 * nb_rollouts), replace=False)
    _obs = [obs[i] for i in _choice]
    _act = [act[i] for i in _choice]

    rarhmm = erARHMM(nb_states, dm_obs, dm_act, type, learn_ctl=False)
    rarhmm.initialize(_obs, _act)

    lls = rarhmm.em(_obs, _act, nb_iter=nb_iter, prec=prec, verbose=False)
    mse, norm_mse = rarhmm.kstep_mse(obs, act, horizon=25, stoch=False)

    return rarhmm, lls, mse, norm_mse


def parallel_em(nb_states, dm_obs, dm_act, type,
                obs, act, nb_iter=50, prec=1e-4, nb_jobs=50):
    args = [(nb_states, dm_obs, dm_act, type, obs, act, nb_iter, prec) for _ in range(nb_jobs)]
    results = Parallel(n_jobs=nb_jobs, verbose=1, backend='loky')(map(delayed(create_job), args))
    rarhmms, lls, mse, norm_mse = list(map(list, zip(*results)))
    return rarhmms, lls, mse, norm_mse


if __name__ == "__main__":

    import os
    import pickle

    import gym
    import rl

    env = gym.make('Pendulum-RL-v0')
    env._max_episode_steps = 5000

    nb_rollouts, nb_steps = 5, 200
    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    reps_ctl = pickle.load(open("reps_pendulum_ctl.pkl", "rb"))
    obs, act = sample_env(env, nb_rollouts, nb_steps)

    nb_states = 5
    models, liklhds, mse, norm_mse = parallel_em(nb_states, dm_obs, dm_act,
                                                 type='neural-recurrent',
                                                 obs=obs, act=act, nb_iter=50,
                                                 prec=1e-4, nb_jobs=10)

    all_models = []
    for _nmse in norm_mse:
        all_models.append(_nmse)
    rarhmm = models[np.argmax(all_models)]

    print(np.c_[mse, norm_mse])

    # path = os.path.dirname(rl.__file__)
    # pickle.dump(erarhmm, open(path + '/envs/control/hybrid/models/hybrid_pendulum.p', 'wb'))
