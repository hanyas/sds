import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import stats

from sds.models import rARHMM
from sds.utils.envs import sample_env

from joblib import Parallel, delayed

import multiprocessing
nb_cores = multiprocessing.cpu_count()


def create_job(kwargs):
    # model arguments
    nb_states = kwargs.pop('nb_states')
    obs_dim = kwargs.pop('obs_dim')
    act_dim = kwargs.pop('act_dim')
    nb_lags = kwargs.pop('nb_lags')

    algo_type = kwargs.pop('algo_type')
    init_obs_type = kwargs.pop('init_obs_type')
    trans_type = kwargs.pop('trans_type')
    obs_type = kwargs.pop('obs_type')

    # model priors
    init_state_prior = kwargs.pop('init_state_prior')
    init_obs_prior = kwargs.pop('init_obs_prior')
    trans_prior = kwargs.pop('trans_prior')
    obs_prior = kwargs.pop('obs_prior')

    # model kwargs
    init_state_kwargs = kwargs.pop('init_state_kwargs')
    init_obs_kwargs = kwargs.pop('init_obs_kwargs')
    trans_kwargs = kwargs.pop('trans_kwargs')
    obs_kwargs = kwargs.pop('obs_kwargs')

    # em arguments
    obs = kwargs.pop('obs')
    act = kwargs.pop('act')

    nb_iter = kwargs.pop('nb_iter')
    prec = kwargs.pop('prec')
    proc_id = kwargs.pop('proc_id')

    init_mstep_kwargs = kwargs.pop('init_mstep_kwargs')
    trans_mstep_kwargs = kwargs.pop('trans_mstep_kwargs')
    obs_mstep_kwargs = kwargs.pop('obs_mstep_kwargs')

    # split train and test data
    from sklearn.model_selection import train_test_split
    list_idx = np.linspace(0, len(obs) - 1, len(obs), dtype=int)
    train_idx, test_idx = train_test_split(list_idx, test_size=0.2, random_state=proc_id)

    train_obs = [obs[i] for i in train_idx]
    train_act = [act[i] for i in train_idx]

    test_obs = [obs[i] for i in test_idx]
    test_act = [act[i] for i in test_idx]

    rarhmm = rARHMM(nb_states=nb_states, obs_dim=obs_dim,
                    act_dim=act_dim, nb_lags=nb_lags,
                    algo_type=algo_type, init_obs_type=init_obs_type,
                    obs_type=obs_type, trans_type=trans_type,
                    init_state_prior=init_state_prior, init_obs_prior=init_obs_prior,
                    trans_prior=trans_prior, obs_prior=obs_prior,
                    init_state_kwargs=init_state_kwargs, init_obs_kwargs=init_obs_kwargs,
                    trans_kwargs=trans_kwargs, obs_kwargs=obs_kwargs)

    rarhmm.em(train_obs, train_act,
              nb_iter=nb_iter, prec=prec,
              initialize=True, proc_id=proc_id,
              init_mstep_kwargs=init_mstep_kwargs,
              trans_mstep_kwargs=trans_mstep_kwargs,
              obs_mstep_kwargs=obs_mstep_kwargs)

    nb_train = np.vstack(train_obs).shape[0]
    nb_all = np.vstack(obs).shape[0]

    train_ll = rarhmm.log_normalizer(train_obs, train_act)
    all_ll = rarhmm.log_normalizer(obs, act)

    train_score = train_ll / nb_train
    test_score = (all_ll - train_ll) / (nb_all - nb_train)

    return rarhmm, train_score, test_score


def parallel_em(nb_jobs=50, **kwargs):
    kwargs_list = []
    for n in range(nb_jobs):
        kwargs['proc_id'] = n
        kwargs_list.append(kwargs.copy())

    results = Parallel(n_jobs=min(nb_jobs, nb_cores),
                       verbose=10, backend='loky')(map(delayed(create_job), kwargs_list))
    rarhmms, train_scores, test_scores = list(map(list, zip(*results)))

    return rarhmms, train_scores, test_scores


if __name__ == "__main__":

    import random
    import torch
    import gym

    random.seed(1337)
    npr.seed(1337)
    torch.manual_seed(1337)
    torch.set_num_threads(1)

    env = gym.make('Cartpole-ID-v1')
    env._max_episode_steps = 5000
    env.unwrapped.dt = 0.01
    env.unwrapped.sigma = 1e-4
    env.seed(1337)

    nb_train_rollouts, nb_train_steps = 15, 250
    nb_test_rollouts, nb_test_steps = 5, 100

    train_obs, train_act = sample_env(env, nb_train_rollouts, nb_train_steps)
    test_obs, test_act = sample_env(env, nb_test_rollouts, nb_test_steps)

    nb_states = 7
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    nb_lags = 1

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
    psi = 1e8 * np.eye(obs_dim) / (obs_dim + 1)
    nu = (obs_dim + 1) + obs_dim + 1

    from sds.distributions.composite import StackedNormalWishart
    init_obs_prior = StackedNormalWishart(nb_states, obs_dim,
                                          mus=np.array([mu for _ in range(nb_states)]),
                                          kappas=np.array([kappa for _ in range(nb_states)]),
                                          psis=np.array([psi for _ in range(nb_states)]),
                                          nus=np.array([nu for _ in range(nb_states)]))

    # obs_prior
    input_dim = obs_dim * nb_lags + act_dim + 1
    output_dim = obs_dim

    M = np.zeros((output_dim, input_dim))
    K = 1e-6 * np.eye(input_dim)
    psi = 1e8 * np.eye(obs_dim) / (obs_dim + 1)
    nu = (obs_dim + 1) + obs_dim + 1

    from sds.distributions.composite import StackedMatrixNormalWishart
    obs_prior = StackedMatrixNormalWishart(nb_states, input_dim, output_dim,
                                           Ms=np.array([M for _ in range(nb_states)]),
                                           Ks=np.array([K for _ in range(nb_states)]),
                                           psis=np.array([psi for _ in range(nb_states)]),
                                           nus=np.array([nu for _ in range(nb_states)]))

    # trans_prior
    trans_prior = {'alpha': 1., 'kappa': 0.5}  # Dirichlet params

    # model kwargs
    init_state_kwargs, init_obs_kwargs, obs_kwargs = {}, {}, {}
    trans_kwargs = {'device': 'cpu',
                    'hidden_sizes': (16,), 'activation': 'splus',
                    'norm': {'mean': np.array([0., 0., 0., 0., 0., 0.]),
                             'std': np.array([5., 1., 1., 5., 10., 5.])}}

    # mstep kwargs
    init_mstep_kwargs, obs_mstep_kwargs = {}, {}
    trans_mstep_kwargs = {'nb_iter': 100, 'batch_size': 64,
                          'lr': 5e-4, 'l2': 1e-32}

    models, train_scores, test_scores = parallel_em(nb_jobs=1, nb_states=nb_states,
                                                    obs_dim=obs_dim, act_dim=act_dim, nb_lags=1,
                                                    algo_type=algo_type, init_obs_type=init_obs_type,
                                                    trans_type=trans_type, obs_type=obs_type,
                                                    init_state_prior=init_state_prior, init_obs_prior=init_obs_prior,
                                                    trans_prior=trans_prior, obs_prior=obs_prior,
                                                    init_state_kwargs=init_state_kwargs, init_obs_kwargs=init_obs_kwargs,
                                                    trans_kwargs=trans_kwargs, obs_kwargs=obs_kwargs,
                                                    obs=train_obs, act=train_act,
                                                    nb_iter=50, prec=1e-4,
                                                    init_mstep_kwargs=init_mstep_kwargs,
                                                    trans_mstep_kwargs=trans_mstep_kwargs,
                                                    obs_mstep_kwargs=obs_mstep_kwargs)

    scores = -1. * np.array([train_scores]) - 1. * np.array([test_scores])
    rarhmm = models[np.argmin(sc.stats.rankdata(scores))]

    print("rarhmm, stochastic, " + rarhmm.trans_type)
    print(np.c_[train_scores, test_scores])

    for i in range(len(train_obs)):
        rarhmm.plot(train_obs[i], train_act[i])

    hr = [1, 5, 10, 15, 20, 25]
    for h in hr:
        mse, smse, evar = rarhmm.kstep_error(test_obs, test_act, horizon=h, average=True)
        print(f"MSE: {mse}, SMSE:{smse}, EVAR:{evar}")
