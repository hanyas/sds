import numpy as np
import numpy.random as npr

from sds.models import EnsembleHiddenMarkovModel
from sds.utils.envs import sample_env

if __name__ == "__main__":

    import random
    import torch
    import gym

    random.seed(1337)
    npr.seed(1337)
    torch.manual_seed(1337)
    torch.set_num_threads(1)

    env = gym.make('Pendulum-ID-v1')
    env._max_episode_steps = 5000
    env.unwrapped.dt = 0.01
    env.unwrapped.sigma = 1e-8
    env.seed(1337)

    nb_train_rollouts, nb_train_steps = 25, 250
    nb_test_rollouts, nb_test_steps = 5, 100

    train_obs, train_act = sample_env(env, nb_train_rollouts, nb_train_steps)
    test_obs, test_act = sample_env(env, nb_test_rollouts, nb_test_steps)

    nb_states = 5
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    obs_lag = 1

    # model types
    algo_type = 'MAP'
    init_obs_type = 'full'
    obs_type = 'full'
    trans_type = 'neural-ensemble'

    # init_state_prior
    init_state_prior = {}

    # init_obs_prior
    mu = np.zeros((obs_dim,))
    kappa = 1e-64
    psi = 1e4 * np.eye(obs_dim) / (obs_dim + 1)
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
    psi = 1e4 * np.eye(obs_dim) / (obs_dim + 1)
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
                    'hidden_sizes': (32,), 'activation': 'splus',
                    'norm': {'mean': np.array([0., 0., 0., 0.]),
                             'std': np.array([1., 1., 10., 2.5])}}

    # mstep kwargs
    init_state_mstep_kwargs = {}
    init_obs_mstep_kwargs = {'method': 'sgd', 'nb_iter': 1, 'lr': 1e-2}
    obs_mstep_kwargs = {'method': 'sgd', 'nb_iter': 1, 'batch_size': 256, 'lr': 1e-2}
    trans_mstep_kwargs = {'nb_iter': 5, 'batch_size': 256, 'lr': 5e-4, 'l2': 1e-32}

    ensemble = EnsembleHiddenMarkovModel(nb_states=nb_states, obs_dim=obs_dim,
                                         act_dim=act_dim, obs_lag=obs_lag,
                                         model_type='rarhmm', ensemble_size=6,
                                         algo_type=algo_type, init_obs_type=init_obs_type,
                                         trans_type=trans_type, obs_type=obs_type,
                                         init_state_prior=init_state_prior,
                                         init_obs_prior=init_obs_prior,
                                         trans_prior=trans_prior, obs_prior=obs_prior,
                                         init_state_kwargs=init_state_kwargs,
                                         init_obs_kwargs=init_obs_kwargs,
                                         trans_kwargs=trans_kwargs, obs_kwargs=obs_kwargs)

    ensemble.em(train_obs, train_act,
                nb_iter=500, tol=1e-4, initialize=True,
                init_state_mstep_kwargs=init_state_mstep_kwargs,
                init_obs_mstep_kwargs=init_obs_mstep_kwargs,
                trans_mstep_kwargs=trans_mstep_kwargs,
                obs_mstep_kwargs=obs_mstep_kwargs)

    hr = [1, 5, 10, 15, 20, 25]
    for h in hr:
        mse, smse, evar = ensemble.kstep_error(test_obs, test_act, horizon=h, average=True)
        print(f"MSE: {mse}, SMSE:{smse}, EVAR:{evar}")

    # import torch
    # torch.save(ensemble, open("ensemble_pendulum_cart.pkl", "wb"))

    hst, hr = 10, 75
    import matplotlib.pyplot as plt
    for obs, act in zip(test_obs, test_act):
        nxt_obs = ensemble.forcast(horizon=hr, hist_obs=obs[:hst, :],
                                   hist_act=act[:hst, :act_dim],
                                   nxt_act=act[hst:, :act_dim], average=True)

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
        for k in range(0, obs_dim):
            axes[k].plot(obs[hst:hst + hr, k], 'b', lw=2)
            axes[k].plot(nxt_obs[:, k], 'r', lw=2)

    plt.show()
