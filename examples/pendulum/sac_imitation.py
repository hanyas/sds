import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import stats

from sds.models import ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel
from sds.utils.envs import sample_env, rollout_policy

from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from matplotlib import rc

import multiprocessing
nb_cores = multiprocessing.cpu_count()


rc('lines', **{'linewidth': 1})
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Palatino']})


def beautify(ax):
    ax.set_frame_on(True)
    ax.minorticks_on()

    ax.grid(True)
    ax.grid(linestyle=':')

    ax.tick_params(which='both', direction='in',
                   bottom=True, labelbottom=True,
                   top=True, labeltop=False,
                   right=True, labelright=False,
                   left=True, labelleft=True)

    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)

    ax.autoscale(tight=True)
    # ax.set_aspect('equal')

    if ax.get_legend():
        ax.legend(loc='best')

    return ax


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
    ctl_type = kwargs.get('ctl_type')

    # model priors
    init_state_prior = kwargs.get('init_state_prior')
    init_obs_prior = kwargs.get('init_obs_prior')
    trans_prior = kwargs.get('trans_prior')
    obs_prior = kwargs.get('obs_prior')
    ctl_prior = kwargs.get('ctl_prior')

    # model kwargs
    init_state_kwargs = kwargs.get('init_state_kwargs')
    init_obs_kwargs = kwargs.get('init_obs_kwargs')
    trans_kwargs = kwargs.get('trans_kwargs')
    obs_kwargs = kwargs.get('obs_kwargs')
    ctl_kwargs = kwargs.get('ctl_kwargs')

    # em arguments
    nb_iter = kwargs.get('nb_iter')
    tol = kwargs.get('tol')
    process_id = seed

    init_mstep_kwargs = kwargs.get('init_state_mstep_kwargs')
    init_mstep_kwargs = kwargs.get('init_obs_mstep_kwargs')
    trans_mstep_kwargs = kwargs.get('trans_mstep_kwargs')
    obs_mstep_kwargs = kwargs.get('obs_mstep_kwargs')
    ctl_mstep_kwargs = kwargs.get('ctl_mstep_kwargs')

    clrarhmm = ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel(nb_states=nb_states, obs_dim=obs_dim, act_dim=act_dim,
                                                                  obs_lag=obs_lag, algo_type=algo_type,
                                                                  init_obs_type=init_obs_type, trans_type=trans_type,
                                                                  obs_type=obs_type, ctl_type=ctl_type,
                                                                  init_state_prior=init_state_prior, init_obs_prior=init_obs_prior,
                                                                  trans_prior=trans_prior, obs_prior=obs_prior, ctl_prior=ctl_prior,
                                                                  init_state_kwargs=init_state_kwargs, init_obs_kwargs=init_obs_kwargs,
                                                                  trans_kwargs=trans_kwargs, obs_kwargs=obs_kwargs, ctl_kwargs=ctl_kwargs)

    clrarhmm.em(train_obs, train_act, nb_iter=nb_iter,
                tol=tol, initialize=True, process_id=process_id,
                init_state_mstep_kwargs=init_state_mstep_kwargs,
                init_obs_mstep_kwargs=init_obs_mstep_kwargs,
                trans_mstep_kwargs=trans_mstep_kwargs,
                obs_mstep_kwargs=obs_mstep_kwargs,
                ctl_mstep_kwargs=ctl_mstep_kwargs)

    return clrarhmm


def parallel_em(train_obs, train_act, **kwargs):

    nb_jobs = len(train_obs)
    kwargs_list = [kwargs.copy() for _ in range(nb_jobs)]
    seeds = np.linspace(0, nb_jobs - 1, nb_jobs, dtype=int)

    clrarhmm = Parallel(n_jobs=min(nb_jobs, nb_cores), verbose=1, backend='loky')\
        (map(delayed(create_job), train_obs, train_act, kwargs_list, seeds))

    return clrarhmm


if __name__ == "__main__":

    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.colors as cls

    sns.set_style("white")
    sns.set_context("talk")

    color_names = ["windows blue", "red", "amber", "faded green", "dusty purple",
                   "greyish", "lightblue", "magenta", "clay", "teal",
                   "marine blue", "orangered", "burnt yellow", "jungle green"]

    colors = sns.xkcd_palette(color_names)
    cmap = cls.ListedColormap(colors)

    import random
    import torch
    import gym

    random.seed(1337)
    npr.seed(1337)
    torch.manual_seed(1337)
    torch.set_num_threads(1)

    env = gym.make('Pendulum-ID-v1')
    env._max_episode_steps = 5000
    env.unwrapped.dt = 0.02
    env.unwrapped.sigma = 1e-4
    env.unwrapped.uniform = True
    env.seed(1337)

    from stable_baselines import SAC

    _ctl = SAC.load("./sac_pendulum")
    sac_ctl = lambda x: _ctl.predict(x)[0]

    nb_rollouts, nb_steps = 50, 200
    obs, act = sample_env(env, nb_rollouts, nb_steps, sac_ctl, noise_std=1e-2)

    from sds.utils.general import train_test_split
    train_obs, train_act, _, _ = train_test_split(obs, act, seed=3,
                                                  nb_traj_splits=6,
                                                  split_trajs=False)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
    fig.suptitle('Pendulum SAC Demonstrations')

    for _obs, _act in zip(obs, act):
        # angle = np.arctan2(_obs[:, 1], _obs[:, 0])
        # axs[0].plot(angle)
        axs[0].plot(_obs[:, 0])
        axs[0] = beautify(axs[0])
        axs[1].plot(_obs[:, -1])
        axs[1] = beautify(axs[1])
        axs[2].plot(_act)
        axs[2] = beautify(axs[2])

    axs[0].set_xlabel('Time Step')
    axs[1].set_xlabel('Time Step')
    axs[2].set_xlabel('Time Step')

    axs[0].set_ylabel('$\\cos(\\theta)$')
    axs[1].set_ylabel('$\\dot{\\theta}$')
    axs[2].set_ylabel('$u$')

    plt.show()

    nb_states = 5
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    obs_lag = 1

    # model types
    algo_type = 'MAP'
    init_obs_type = 'full'
    obs_type = 'full'
    trans_type = 'neural'
    ctl_type = 'full'

    ctl_degree = 3

    # init_state_prior
    from sds.distributions.dirichlet import Dirichlet
    init_state_prior = Dirichlet(dim=nb_states, alphas=np.ones((nb_states, )))

    # init_obs_prior
    mu = np.zeros((obs_dim,))
    kappa = 1e-64
    psi = 1e4 * np.eye(obs_dim) / (obs_dim + 1)
    nu = (obs_dim + 1) + obs_dim + 1
    # psi = np.eye(obs_dim)
    # nu = (obs_dim + 1) + 1e-8

    from sds.distributions.composite import StackedNormalWisharts
    init_obs_prior = StackedNormalWisharts(nb_states, obs_dim,
                                           mus=np.array([mu for _ in range(nb_states)]),
                                           kappas=np.array([kappa for _ in range(nb_states)]),
                                           psis=np.array([psi for _ in range(nb_states)]),
                                           nus=np.array([nu for _ in range(nb_states)]))

    # trans_prior
    trans_prior = {'alpha': 1., 'kappa': 0.}  # Dirichlet params

    # obs_prior
    input_dim = obs_dim * obs_lag + act_dim + 1
    output_dim = obs_dim

    M = np.zeros((output_dim, input_dim))
    K = 1e-6 * np.eye(input_dim)
    psi = 1e4 * np.eye(output_dim) / (output_dim + 1)
    nu = (output_dim + 1) + output_dim + 1
    # psi = np.eye(output_dim)
    # nu = (output_dim + 1) + 1e-8

    from sds.distributions.composite import TiedMatrixNormalWisharts
    obs_prior = TiedMatrixNormalWisharts(nb_states, input_dim, output_dim,
                                         Ms=np.array([M for _ in range(nb_states)]),
                                         Ks=np.array([K for _ in range(nb_states)]),
                                         psis=np.array([psi for _ in range(nb_states)]),
                                         nus=np.array([nu for _ in range(nb_states)]))

    # ctl_prior
    from scipy import special
    feat_dim = int(sc.special.comb(ctl_degree + obs_dim, ctl_degree)) - 1
    input_dim = feat_dim + 1
    output_dim = act_dim

    M = np.zeros((output_dim, input_dim))
    K = 1e-6 * np.eye(input_dim)
    # psi = 1e2 * np.eye(act_dim) / (act_dim + 1)
    # nu = (act_dim + 1) + act_dim + 1
    psi = np.eye(act_dim)
    nu = (act_dim + 1) + 1e-8

    from sds.distributions.composite import StackedMatrixNormalWisharts
    ctl_prior = StackedMatrixNormalWisharts(nb_states, input_dim, output_dim,
                                            Ms=np.array([M for _ in range(nb_states)]),
                                            Ks=np.array([K for _ in range(nb_states)]),
                                            psis=np.array([psi for _ in range(nb_states)]),
                                            nus=np.array([nu for _ in range(nb_states)]))

    # model kwargs
    init_state_kwargs, init_obs_kwargs = {}, {}
    obs_kwargs, ctl_kwargs = {}, {'degree': ctl_degree}
    trans_kwargs = {'device': 'cpu',
                    'hidden_sizes': (32,), 'activation': 'splus',
                    'norm': {'mean': np.array([0., 0., 0., 0.]),
                             'std': np.array([1., 1., 10., 2.5])}}

    # mstep kwargs
    init_state_mstep_kwargs = {}
    init_obs_mstep_kwargs = {}
    obs_mstep_kwargs = {}
    ctl_mstep_kwargs = {}
    trans_mstep_kwargs = {'nb_iter': 5, 'batch_size': 512, 'lr': 5e-4, 'l2': 1e-32}

    models = parallel_em(train_obs=train_obs, train_act=train_act,
                         nb_states=nb_states, obs_dim=obs_dim,
                         act_dim=act_dim, obs_lag=obs_lag,
                         algo_type=algo_type, init_obs_type=init_obs_type,
                         trans_type=trans_type, obs_type=obs_type, ctl_type=ctl_type,
                         init_state_prior=init_state_prior, init_obs_prior=init_obs_prior,
                         trans_prior=trans_prior, obs_prior=obs_prior, ctl_prior=ctl_prior,
                         init_state_kwargs=init_state_kwargs, init_obs_kwargs=init_obs_kwargs,
                         trans_kwargs=trans_kwargs, obs_kwargs=obs_kwargs, ctl_kwargs=ctl_kwargs,
                         nb_iter=100, tol=1e-4,
                         init_state_mstep_kwargs=init_state_mstep_kwargs,
                         init_obs_mstep_kwargs=init_obs_mstep_kwargs,
                         trans_mstep_kwargs=trans_mstep_kwargs,
                         obs_mstep_kwargs=obs_mstep_kwargs,
                         ctl_mstep_kwargs=ctl_mstep_kwargs)

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
    clrarhmm = models[np.argmin(sc.stats.rankdata(-1. * scores))]

    # ctl = clrarhmm.smoothed_control(obs, act)
    # # state, ctl = clrarhmm.filtered_control(obs, act)
    #
    # fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 12), constrained_layout=True)
    # fig.suptitle('Demonstration Action Filtering')
    #
    # idx = npr.choice(len(obs))
    #
    # angle = np.arctan2(obs[idx][:, 1], obs[idx][:, 0])
    # axs[0].plot(angle)
    # axs[0].set_ylabel('$\\theta$')
    # axs[0].set_xlim(0, len(obs[idx]))
    #
    # axs[1].plot(obs[idx][:, -1], '-g')
    # axs[1].set_ylabel("$\\dot{\\theta}$")
    # axs[1].set_xlim(0, len(obs[idx]))
    #
    # axs[2].plot(act[idx])
    # axs[2].plot(ctl[idx])
    # axs[2].legend(('Actual', 'Smoothed'))
    # axs[2].set_ylabel("$u$")
    # axs[2].set_xlim(0, len(act[idx]))
    #
    # # axs[3].imshow(state[idx][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    # # axs[3].set_xlim(0, len(obs[idx]))
    # # axs[3].set_xlabel('Time Step')
    # # axs[3].set_ylabel("$z_{\\mathrm{inferred}}$")
    # # axs[3].set_yticks([])
    #
    # plt.show()

    rollouts = rollout_policy(env, clrarhmm, 50, 250, average=True, stoch=True)

    # fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 12), constrained_layout=True)
    # fig.suptitle('Pendulum Hybrid Imitation: One Example')
    #
    # idx = np.random.choice(len(rollouts))
    #
    # # angle = np.arctan2(rollouts[idx]['x'][:, 1], rollouts[idx]['x'][:, 0])
    # # axs[0].plot(angle)
    # # axs[0].set_ylabel('$\\theta$')
    # # axs[0].set_xlim(0, len(rollouts[idx]['x']))
    #
    # axs[0].plot(rollouts[idx]['x'][:, :-1])
    # axs[0].set_ylabel('$\\cos(\\theta)/\\sin(\\theta)$')
    # axs[0].set_xlim(0, len(rollouts[idx]['x']))
    #
    # axs[1].plot(rollouts[idx]['x'][:, -1], '-g')
    # axs[1].set_ylabel("$\\dot{\\theta}$")
    # axs[1].set_xlim(0, len(rollouts[idx]['x']))
    #
    # axs[2].plot(rollouts[idx]['u'], '-r')
    # axs[2].set_ylabel('$u$')
    # axs[2].set_xlim(0, len(rollouts[idx]['u']))
    #
    # axs[3].imshow(rollouts[idx]['z'][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    # axs[3].set_xlim(0, len(rollouts[idx]['z']))
    # axs[3].set_xlabel('Time Step')
    # axs[3].set_ylabel("$z_{\\mathrm{inferred}}$")
    # axs[3].set_yticks([])
    #
    # plt.show()

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 6), constrained_layout=True)
    fig.suptitle('Pendulum Hybrid Imitation: Many Seeds')

    for roll in rollouts:
        axs[0].plot(roll['x'][:, 0])
        axs[0] = beautify(axs[0])
        axs[1].plot(roll['x'][:, -1])
        axs[1] = beautify(axs[1])
        axs[2].plot(roll['u'])
        axs[2] = beautify(axs[2])
        axs[3].plot(roll['z'])
        axs[3] = beautify(axs[3])

    axs[0].set_xlabel('Time Step')
    axs[1].set_xlabel('Time Step')
    axs[2].set_xlabel('Time Step')
    axs[3].set_xlabel('Time Step')

    axs[0].set_ylabel('$\\cos(\\theta)$')
    axs[1].set_ylabel("$\\dot{\\theta}$")
    axs[2].set_ylabel('$u$')
    axs[3].set_ylabel('$z$')

    plt.show()

    # phase portraits
    def ang2cart(x):
        if x.ndim == 1:
            state = np.zeros((3,))
            state[0] = np.cos(x[0])
            state[1] = np.sin(x[0])
            state[2] = x[1]
            return state
        return np.vstack(list(map(ang2cart, list(x))))

    xlim = (-np.pi, np.pi)
    ylim = (-8.0, 8.0)

    npts = 36
    x = np.linspace(*xlim, npts)
    y = np.linspace(*ylim, npts)

    X, Y = np.meshgrid(x, y)
    XYi = np.stack((X, Y))
    XYn = np.zeros((2, npts, npts))

    # SAC closed-loop
    env.reset()
    for i in range(npts):
        for j in range(npts):
            _u = sac_ctl(ang2cart(XYi[:, i, j]))
            XYn[:, i, j] = env.unwrapped.fake_step(XYi[:, i, j], _u)

    dXY = XYn - XYi

    fig = plt.figure(figsize=(5, 5), frameon=True)
    ax = fig.gca()

    ax.streamplot(x, y, dXY[0, ...], dXY[1, ...],
                  color='g', linewidth=1, density=1.25,
                  arrowstyle='-|>', arrowsize=1.,
                  minlength=0.25)

    ax = beautify(ax)
    ax.grid(False)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.show()

    # collect env history
    hr = 3
    XYh = np.zeros((hr, 2, npts, npts))

    env.reset()
    for i in range(npts):
        for j in range(npts):
            XYh[0, :, i, j] = XYi[:, i, j]
            for t in range(1, hr):
                XYh[t, :, i, j] = env.unwrapped.fake_step(XYh[t - 1, :, i, j], np.array([0.0]))

    # hybrid closed-loop
    env.reset()
    for i in range(npts):
        for j in range(npts):
            hist_obs, hist_act = ang2cart(XYh[..., i, j]), np.zeros((hr, act_dim))
            _, _, u = clrarhmm.action(hist_obs, hist_act)
            XYn[:, i, j] = env.unwrapped.fake_step(XYh[-1, :, i, j], u)

    dXY = XYn - XYh[-1, ...]

    # re-interpolate data for streamplot
    xh, yh = XYh[-1, 0, 1, :], XYh[-1, 1, :, 0]
    xi = np.linspace(xh.min(), xh.max(), x.size)
    yi = np.linspace(yh.min(), yh.max(), y.size)

    from scipy.interpolate import interp2d

    dxh, dyh = dXY[0, ...], dXY[1, ...]
    dxi = interp2d(xh, yh, dxh)(xi, yi)
    dyi = interp2d(xh, yh, dyh)(xi, yi)

    fig = plt.figure(figsize=(5, 5), frameon=True)
    ax = fig.gca()

    ax.streamplot(xi, yi, dxi, dyi,
                  color='r', linewidth=1, density=1.25,
                  arrowstyle='-|>', arrowsize=1.,
                  minlength=0.25)

    ax = beautify(ax)
    ax.grid(False)

    ax.set_xlim((xh.min(), xh.max()))
    ax.set_ylim((yh.min(), yh.max()))

    plt.show()

    # # from tikzplotlib import save
    # save("sac_pendulum_imitation_phase.tex")

    # success rate
    success = 0.
    for roll in rollouts:
        angle = np.arctan2(roll['x'][:, 1], roll['x'][:, 0])
        if np.all(np.fabs(angle[200:]) < np.deg2rad(15)):
            success += 1.

    print('Imitation Success Rate: ', success / len(rollouts))
