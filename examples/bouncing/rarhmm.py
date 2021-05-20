import numpy as np
import numpy.random as npr

from sds.models import rARHMM
from sds.utils.envs import sample_env
import matplotlib.pyplot as plt


if __name__ == "__main__":

    import random
    import torch
    import gym

    random.seed(1337)
    npr.seed(1337)
    torch.manual_seed(1337)

    env = gym.make('BouncingBall-ID-v0')
    env._max_episode_steps = 5000
    env.unwrapped.dt = 0.05
    env.unwrapped.sigma = 1e-16
    env.seed(1337)

    nb_rollouts, nb_steps = 25, 150

    obs, act = sample_env(env, nb_rollouts, nb_steps)

    nb_states = 2
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
    trans_prior = {'alpha': 1., 'kappa': 0.0}  # Dirichlet params

    # model kwargs
    init_state_kwargs, init_obs_kwargs, obs_kwargs = {}, {}, {}
    trans_kwargs = {'device': 'cpu',
                    'hidden_sizes': (16,), 'activation': 'splus'}

    # mstep kwargs
    init_mstep_kwargs, obs_mstep_kwargs = {}, {}
    trans_mstep_kwargs = {'nb_iter': 25, 'batch_size': 256,
                          'lr': 1e-3, 'l2': 1e-32}

    rarhmm = rARHMM(nb_states=nb_states, obs_dim=obs_dim,
                    act_dim=act_dim, nb_lags=nb_lags,
                    algo_type=algo_type, init_obs_type=init_obs_type,
                    obs_type=obs_type, trans_type=trans_type,
                    init_state_prior=init_state_prior, init_obs_prior=init_obs_prior,
                    trans_prior=trans_prior, obs_prior=obs_prior,
                    init_state_kwargs=init_state_kwargs, init_obs_kwargs=init_obs_kwargs,
                    trans_kwargs=trans_kwargs, obs_kwargs=obs_kwargs)

    rarhmm.em(obs, act,
              nb_iter=100, prec=1e-4,
              initialize=True,
              init_mstep_kwargs=init_mstep_kwargs,
              trans_mstep_kwargs=trans_mstep_kwargs,
              obs_mstep_kwargs=obs_mstep_kwargs)

    fig, ax = plt.subplots(nrows=1, ncols=obs_dim + act_dim, figsize=(12, 4))
    for _obs, _act in zip(obs, act):
        for k, col in enumerate(ax[:-1]):
            col.plot(_obs[:, k])
        ax[-1].plot(_act)
    plt.show()

    seq = npr.choice(len(obs))
    rarhmm.plot(obs[seq], act[seq])

    hr = [20, 40, 60, 80, 100]
    for h in hr:
        print(rarhmm.kstep_error(obs[0:5], act[0:5], horizon=h))
