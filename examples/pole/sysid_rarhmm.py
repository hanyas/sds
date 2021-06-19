import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import stats

from sds.models import RecurrentAutoRegressiveHiddenMarkovModel
from sds.utils.envs import sample_env

from joblib import Parallel, delayed

import multiprocessing
nb_cores = multiprocessing.cpu_count()


def create_job(train_obs, train_act, kwargs, seed):

    random.seed(seed)
    npr.seed(seed)
    torch.manual_seed(seed)

    # model arguments
    nb_states = kwargs.get('nb_states')
    obs_dim = kwargs.get('obs_dim')
    act_dim = kwargs.get('act_dim')
    obs_lag = kwargs.get('obs_lag')

    algo_type = kwargs.get('algo_type')
    init_obs_type = kwargs.get('init_obs_type')
    trans_type = kwargs.get('trans_type')
    obs_type = kwargs.get('obs_type')

    # model priors
    init_state_prior = kwargs.get('init_state_prior')
    init_obs_prior = kwargs.get('init_obs_prior')
    trans_prior = kwargs.get('trans_prior')
    obs_prior = kwargs.get('obs_prior')

    # model kwargs
    init_state_kwargs = kwargs.get('init_state_kwargs')
    init_obs_kwargs = kwargs.get('init_obs_kwargs')
    trans_kwargs = kwargs.get('trans_kwargs')
    obs_kwargs = kwargs.get('obs_kwargs')

    # em arguments
    nb_iter = kwargs.get('nb_iter')
    prec = kwargs.get('prec')
    proc_id = seed

    init_mstep_kwargs = kwargs.get('init_state_mstep_kwargs')
    init_mstep_kwargs = kwargs.get('init_obs_mstep_kwargs')
    trans_mstep_kwargs = kwargs.get('trans_mstep_kwargs')
    obs_mstep_kwargs = kwargs.get('obs_mstep_kwargs')

    rarhmm = RecurrentAutoRegressiveHiddenMarkovModel(nb_states=nb_states, obs_dim=obs_dim,
                                                      act_dim=act_dim, obs_lag=obs_lag,
                                                      algo_type=algo_type, init_obs_type=init_obs_type,
                                                      trans_type=trans_type, obs_type=obs_type,
                                                      init_state_prior=init_state_prior, init_obs_prior=init_obs_prior,
                                                      trans_prior=trans_prior, obs_prior=obs_prior,
                                                      init_state_kwargs=init_state_kwargs, init_obs_kwargs=init_obs_kwargs,
                                                      trans_kwargs=trans_kwargs, obs_kwargs=obs_kwargs)

    rarhmm.em(train_obs, train_act,
              nb_iter=nb_iter, prec=prec,
              initialize=True, proc_id=proc_id,
              init_state_mstep_kwargs=init_state_mstep_kwargs,
              init_obs_mstep_kwargs=init_obs_mstep_kwargs,
              trans_mstep_kwargs=trans_mstep_kwargs,
              obs_mstep_kwargs=obs_mstep_kwargs)

    return rarhmm


def parallel_em(train_obs, train_act, **kwargs):

    nb_jobs = len(train_obs)
    kwargs_list = [kwargs.copy() for _ in range(nb_jobs)]
    seeds = np.linspace(0, nb_jobs - 1, nb_jobs, dtype=int)

    rarhmms = Parallel(n_jobs=min(nb_jobs, nb_cores), verbose=1, backend='loky')\
        (map(delayed(create_job), train_obs, train_act, kwargs_list, seeds))

    return rarhmms


if __name__ == "__main__":

    import random
    import torch
    import gym

    random.seed(1337)
    npr.seed(1337)
    torch.manual_seed(1337)
    torch.set_num_threads(1)

    env = gym.make('Pole-ID-v0')
    env._max_episode_steps = 5000
    env.unwrapped.dt = 0.01
    env.unwrapped.sigma = 1e-8
    env.unwrapped.uniform = True
    env.seed(1337)

    nb_train_rollouts, nb_train_steps = 25, 25
    nb_test_rollouts, nb_test_steps = 5, 25

    obs, act = sample_env(env, nb_train_rollouts, nb_train_steps)
    test_obs, test_act = sample_env(env, nb_test_rollouts, nb_test_steps)

    from sds.utils.general import train_test_split
    train_obs, train_act, _, _ = train_test_split(obs, act, seed=1337,
                                                  nb_traj_splits=6,
                                                  split_trajs=False)

    nb_states = 2
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    obs_lag = 1

    # model types
    algo_type = 'MAP'
    init_obs_type = 'full'
    obs_type = 'full'
    trans_type = 'neural'

    # init_state_prior
    init_state_prior = {}

    # init_obs_prior
    mu = np.zeros((obs_dim,))
    kappa = 1e-64
    psi = 1e2 * np.eye(obs_dim) / (obs_dim + 1)
    nu = (obs_dim + 1) + obs_dim + 1
    # psi = np.eye(obs_dim)
    # nu = obs_dim + 1 + 1e-8

    from sds.distributions.composite import StackedNormalWishart
    init_obs_prior = StackedNormalWishart(nb_states, obs_dim,
                                          mus=np.array([mu for _ in range(nb_states)]),
                                          kappas=np.array([kappa for _ in range(nb_states)]),
                                          psis=np.array([psi for _ in range(nb_states)]),
                                          nus=np.array([nu for _ in range(nb_states)]))

    # obs_prior
    input_dim = obs_dim * obs_lag + act_dim + 1
    output_dim = obs_dim

    M = np.zeros((output_dim, input_dim))
    K = 1e-6 * np.eye(input_dim)
    psi = 1e2 * np.eye(obs_dim) / (obs_dim + 1)
    nu = (obs_dim + 1) + obs_dim + 1
    # psi = np.eye(obs_dim)
    # nu = obs_dim + 1 + 1e-8

    from sds.distributions.composite import StackedMatrixNormalWishart
    obs_prior = StackedMatrixNormalWishart(nb_states, input_dim, output_dim,
                                           Ms=np.array([M for _ in range(nb_states)]),
                                           Ks=np.array([K for _ in range(nb_states)]),
                                           psis=np.array([psi for _ in range(nb_states)]),
                                           nus=np.array([nu for _ in range(nb_states)]))

    # trans_prior
    trans_prior = {'alpha': 1., 'kappa': 0.}  # Dirichlet params

    # model kwargs
    init_state_kwargs, init_obs_kwargs, obs_kwargs = {}, {}, {}
    trans_kwargs = {'device': 'cpu',
                    'hidden_sizes': (8,), 'activation': 'relu'}

    # mstep kwargs
    init_state_mstep_kwargs = {}
    init_obs_mstep_kwargs = {'method': 'sgd', 'nb_iter': 1, 'lr': 1e-2}
    obs_mstep_kwargs = {'method': 'sgd', 'nb_iter': 1, 'batch_size': 32, 'lr': 1e-2}
    trans_mstep_kwargs = {'nb_iter': 5, 'batch_size': 32, 'lr': 5e-4, 'l2': 1e-32}

    models = parallel_em(train_obs=train_obs, train_act=train_act,
                         nb_states=nb_states, obs_dim=obs_dim,
                         act_dim=act_dim, obs_lag=obs_lag,
                         algo_type=algo_type, init_obs_type=init_obs_type,
                         trans_type=trans_type, obs_type=obs_type,
                         init_state_prior=init_state_prior, init_obs_prior=init_obs_prior,
                         trans_prior=trans_prior, obs_prior=obs_prior,
                         init_state_kwargs=init_state_kwargs, init_obs_kwargs=init_obs_kwargs,
                         trans_kwargs=trans_kwargs, obs_kwargs=obs_kwargs,
                         nb_iter=500, prec=1e-4,
                         init_state_mstep_kwargs=init_state_mstep_kwargs,
                         init_obs_mstep_kwargs=init_obs_mstep_kwargs,
                         trans_mstep_kwargs=trans_mstep_kwargs,
                         obs_mstep_kwargs=obs_mstep_kwargs)

    # model validation
    nb_train = [np.vstack(x).shape[0] for x in train_obs]
    nb_total = np.vstack(obs).shape[0]

    train_ll, total_ll = [], []
    for x, u, m in zip(train_obs, train_act, models):
        train_ll.append(m.log_normalizer(x, u))
        total_ll.append(m.log_normalizer(obs, act))

    train_scores = np.hstack(train_ll) / np.hstack(nb_train)
    test_scores = (np.hstack(total_ll) - np.hstack(train_ll)) \
                  / (nb_total - np.hstack(nb_train))

    scores = np.array([train_scores]) + np.array([test_scores])
    rarhmm = models[np.argmin(sc.stats.rankdata(-1. * scores))]

    # # plot trajectories
    # for i in range(len(obs)):
    #     rarhmm.plot(obs[i], act[i])

    # validate model on test set
    hr = [1, 5, 10, 15, 20]
    for h in hr:
        mse, smse, evar = rarhmm.kstep_error(test_obs, test_act, horizon=h, average=True)
        print(f"MSE: {mse}, SMSE:{smse}, EVAR:{evar}")

    import torch
    torch.save(rarhmm, open("../../sds/envs/hybrid/models/rarhmm_pole.pkl", "wb"))
