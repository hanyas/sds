import torch
import autograd.numpy as np
import autograd.numpy.random as npr

from sds import rARHMM
from sds.utils import sample_env

from reg.gp import DynamicMultiGPRegressor
from reg.nn import DynamicNNRegressor
from reg.nn import DynamicRNNRegressor
from reg.nn import DynamicLSTMRegressor

import random
from joblib import Parallel, delayed

from matplotlib import rc

to_float = lambda arr: torch.from_numpy(arr).float()
to_double = lambda arr: torch.from_numpy(arr).double()


rc('lines', **{'linewidth': 1})
rc('text', usetex=True)


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


def fit_rarhmm(obs, act, nb_states):
    dm_obs = obs[0].shape[-1]
    dm_act = act[0].shape[-1]

    obs_prior = {'mu0': 0., 'sigma0': 1e64, 'nu0': (dm_obs + 1) + 10, 'psi0': 1e-8 * 10}
    obs_mstep_kwargs = {'use_prior': True}

    trans_type = 'neural'
    trans_prior = {'l2_penalty': 0., 'alpha': 1, 'kappa': 5}
    trans_kwargs = {'hidden_layer_sizes': (25,),
                    'norm': {'mean': np.array([0., 0., 0., 0., 0., 0.]),
                             'std': np.array([5., 1., 1., 5., 10., 5.])}}
    trans_mstep_kwargs = {'nb_iter': 25, 'batch_size': 128, 'lr': 1e-4}

    rarhmm = rARHMM(nb_states, dm_obs, dm_act,
                    trans_type=trans_type,
                    obs_prior=obs_prior,
                    trans_prior=trans_prior,
                    trans_kwargs=trans_kwargs)
    # rarhmm.initialize(obs, act)

    rarhmm.em(obs=obs, act=act,
              nb_iter=500, prec=1e-4, verbose=True,
              obs_mstep_kwargs=obs_mstep_kwargs,
              trans_mstep_kwargs=trans_mstep_kwargs)

    return rarhmm


def test_rarhmm(model, obs, act, horizon):
    mse, evar = model.kstep_mse(obs, act, horizon=horizon, mix=False)
    return mse, evar


def fit_rarhmm_job(args):
    obs, act, nb_states = args

    nb_rollouts = len(obs)
    choice = npr.choice(nb_rollouts, size=int(0.8 * nb_rollouts), replace=False)

    _obs = [obs[i] for i in choice]
    _act = [act[i] for i in choice]
    rarhmm = fit_rarhmm(_obs, _act, nb_states)

    return rarhmm


def test_rarhmm_job(args):
    model, obs, act, horizon = args
    return test_rarhmm(model, obs, act, horizon)


def parallel_rarhmm_fit(obs, act, nb_states, nb_jobs):
    args = [(obs, act, nb_states) for _ in range(nb_jobs)]
    return Parallel(nb_jobs, verbose=10, backend='loky')(map(delayed(fit_rarhmm_job), args))


def parallel_rarhmm_test(models, obs, act, horizon):
    args = [(model, obs, act, horizon) for model in models]
    results = Parallel(len(models), verbose=10, backend='loky')(map(delayed(test_rarhmm_job), args))
    mse, evar = list(map(list, zip(*results)))
    return mse, evar


def fit_gp(obs, act, nb_iter=75):
    input = np.vstack([np.hstack((_x[:-1, :], _u[:-1, :])) for _x, _u in zip(obs, act)])
    target = np.vstack([_x[1:, :] - _x[:-1, :] for _x in obs])

    gp = DynamicMultiGPRegressor(to_float(input), to_float(target))
    gp.fit(nb_iter)

    return gp


def test_gp(model, obs, act, horizon):
    state = [to_float(_x[:-1, :]) for _x in obs]
    action = [to_float(_u[:-1, :]) for _u in act]
    target = [to_float(_x[1:, :]) for _x in obs]

    mse, evar = model.kstep_mse(target, state, action, horizon)
    return mse, evar


def fit_gp_job(args):
    obs, act, nb_iter = args

    nb_rollouts = len(obs)
    choice = npr.choice(nb_rollouts, size=int(0.8 * nb_rollouts), replace=False)

    _obs = [obs[i] for i in choice]
    _act = [act[i] for i in choice]
    gp = fit_gp(_obs, _act, nb_iter)

    return gp


def test_gp_job(args):
    model, obs, act, horizon = args
    return test_gp(model, obs, act, horizon)


def parallel_gp_fit(obs, act, nb_iter, nb_jobs):
    args = [(obs, act, nb_iter) for _ in range(nb_jobs)]
    return Parallel(n_jobs=nb_jobs, verbose=10, backend='loky')(map(delayed(fit_gp_job), args))


def parallel_gp_test(models, obs, act, horizon):
    args = [(model, obs, act, horizon) for model in models]
    results = Parallel(len(models), verbose=10, backend='loky')(map(delayed(test_gp_job), args))
    mse, evar = list(map(list, zip(*results)))
    return mse, evar


def fit_fnn(obs, act, nb_epochs=5000):
    input = np.vstack([np.hstack((_x[:-1, :], _u[:-1, :])) for _x, _u in zip(obs, act)])
    target = np.vstack([_x[1:, :] - _x[:-1, :] for _x in obs])

    fnn = DynamicNNRegressor([input.shape[-1], 16, 16, target.shape[-1]])
    fnn.fit(to_float(target), to_float(input), nb_epochs, batch_size=64)

    return fnn


def test_fnn(model, obs, act, horizon):
    state = [to_float(_x[:-1, :]) for _x in obs]
    action = [to_float(_u[:-1, :]) for _u in act]
    target = [to_float(_x[1:, :]) for _x in obs]

    mse, evar = model.kstep_mse(target, state, action, horizon)
    return mse, evar


def fit_fnn_job(args):
    obs, act, nb_epochs = args

    nb_rollouts = len(obs)
    choice = npr.choice(nb_rollouts, size=int(0.8 * nb_rollouts), replace=False)

    _obs = [obs[i] for i in choice]
    _act = [act[i] for i in choice]
    fnn = fit_fnn(_obs, _act, nb_epochs)

    return fnn


def test_fnn_job(args):
    model, obs, act, horizon = args
    return test_fnn(model, obs, act, horizon)


def parallel_fnn_fit(obs, act, nb_epochs, nb_jobs):
    args = [(obs, act, nb_epochs) for _ in range(nb_jobs)]
    return Parallel(n_jobs=nb_jobs, verbose=10, backend='loky')(map(delayed(fit_fnn_job), args))


def parallel_fnn_test(models, obs, act, horizon):
    args = [(model, obs, act, horizon) for model in models]
    results = Parallel(len(models), verbose=10, backend='loky')(map(delayed(test_fnn_job), args))
    mse, evar = list(map(list, zip(*results)))
    return mse, evar


def fit_rnn(obs, act, nb_epochs=5000):
    input = np.stack((np.hstack((_obs[:-1, :], _act[:-1, :]))
                      for _obs, _act in zip(obs, act)), axis=0)
    target = np.stack((_obs[1:, :] for _obs in obs), axis=0)

    input_size = input.shape[-1]
    target_size = target.shape[-1]

    rnn = DynamicRNNRegressor(input_size, target_size, hidden_size=16, nb_layers=2)
    rnn.fit(to_float(target), to_float(input), nb_epochs)

    return rnn


def test_rnn(model, obs, act, horizon):
    state = [to_float(_x[:-1, :]) for _x in obs]
    action = [to_float(_u[:-1, :]) for _u in act]
    target = [to_float(_x[1:, :]) for _x in obs]

    mse, evar = model.kstep_mse(target, state, action, horizon)
    return mse, evar


def fit_rnn_job(args):
    obs, act,  nb_epochs = args

    nb_rollouts = len(obs)
    choice = npr.choice(nb_rollouts, size=int(0.8 * nb_rollouts), replace=False)

    _obs = [obs[i] for i in choice]
    _act = [act[i] for i in choice]
    rnn = fit_fnn(_obs, _act, nb_epochs)

    return rnn


def test_rnn_job(args):
    model, obs, act, horizon = args
    return test_rnn(model, obs, act, horizon)


def parallel_rnn_fit(obs, act, nb_epochs, nb_jobs):
    args = [(obs, act, nb_epochs) for _ in range(nb_jobs)]
    return Parallel(n_jobs=nb_jobs, verbose=10, backend='loky')(map(delayed(fit_rnn_job), args))


def parallel_rnn_test(models, obs, act, horizon):
    args = [(model, obs, act, horizon) for model in models]
    results = Parallel(len(models), verbose=10, backend='loky')(map(delayed(test_rnn_job), args))
    mse, evar = list(map(list, zip(*results)))
    return mse, evar


def fit_lstm(obs, act, nb_epochs=100):
    input = np.stack((np.hstack((_obs[:-1, :], _act[:-1, :]))
                      for _obs, _act in zip(obs, act)), axis=0)
    target = np.stack((_obs[1:, :] for _obs in obs), axis=0)

    input_size = input.shape[-1]
    target_size = target.shape[-1]

    lstm = DynamicLSTMRegressor(input_size, target_size, [16, 16])
    lstm.fit(to_double(target), to_double(input), nb_epochs, lr=0.1)

    return lstm


def test_lstm(model, obs, act, horizon):
    state = [to_double(_x[:-1, :]) for _x in obs]
    action = [to_double(_u[:-1, :]) for _u in act]
    target = [to_double(_x[1:, :]) for _x in obs]

    mse, evar = model.kstep_mse(target, state, action, horizon)
    return mse, evar


def fit_lstm_job(args):
    obs, act,  nb_epochs = args

    nb_rollouts = len(obs)
    choice = npr.choice(nb_rollouts, size=int(0.8 * nb_rollouts), replace=False)

    _obs = [obs[i] for i in choice]
    _act = [act[i] for i in choice]
    lstm = fit_lstm(_obs, _act, nb_epochs)

    return lstm


def test_lstm_job(args):
    model, obs, act, horizon = args
    return test_lstm(model, obs, act, horizon)


def parallel_lstm_fit(obs, act, nb_epochs, nb_jobs):
    args = [(obs, act, nb_epochs) for _ in range(nb_jobs)]
    return Parallel(n_jobs=nb_jobs, verbose=10, backend='loky')(map(delayed(fit_lstm_job), args))


def parallel_lstm_test(models, obs, act, horizon):
    args = [(model, obs, act, horizon) for model in models]
    results = Parallel(len(models), verbose=10, backend='loky')(map(delayed(test_lstm_job), args))
    mse, evar = list(map(list, zip(*results)))
    return mse, evar


if __name__ == "__main__":

    import os
    import argparse

    import gym
    import sds

    parser = argparse.ArgumentParser(description='Compare SOTA Models on Cartpole')
    parser.add_argument('--model', help='Choose model', default='rarhmm')
    args = parser.parse_args()

    random.seed(1337)
    npr.seed(1337)
    torch.manual_seed(1337)

    env = gym.make('Cartpole-ID-v1')
    env._max_episode_steps = 5000
    env.unwrapped._dt = 0.01
    env.unwrapped._sigma = 1e-8
    env.seed(1337)

    nb_train_rollouts, nb_train_steps = 25, 250
    nb_test_rollouts, nb_test_steps = 5, 250

    hr = [1, 5, 10, 15, 20, 25]

    train_obs, train_act = sample_env(env, nb_train_rollouts, nb_train_steps)
    test_obs, test_act = sample_env(env, nb_test_rollouts, nb_test_steps)

    k, k_mse, k_evar = [], [], []
    if args.model == 'rarhmm':
        # fit rarhmm
        rarhmms = parallel_rarhmm_fit(obs=train_obs, act=train_act,
                                      nb_states=5, nb_jobs=25)
        for h in hr:
            print("Horizon: ", h)
            mse, evar = parallel_rarhmm_test(rarhmms, test_obs, test_act, int(h))

            k_mse.append(np.min(mse))
            k_evar.append(np.max(evar))
            k.append(h)

    elif args.model == 'gp':
        # fit gp
        gps = parallel_gp_fit(obs=train_obs, act=train_act,
                              nb_iter=75, nb_jobs=1)
        for h in hr:
            print("Horizon: ", h)
            mse, evar = parallel_gp_test(gps, test_obs, test_act, int(h))

            k_mse.append(np.min(mse))
            k_evar.append(np.max(evar))
            k.append(h)

    elif args.model == 'fnn':
        # fit fnn
        fnns = parallel_fnn_fit(obs=train_obs, act=train_act,
                                nb_epochs=7500, nb_jobs=25)
        for h in hr:
            print("Horizon: ", h)
            mse, evar = parallel_fnn_test(fnns, test_obs, test_act, int(h))

            k_mse.append(np.min(mse))
            k_evar.append(np.max(evar))
            k.append(h)

    elif args.model == 'rnn':
        # fit rnn
        rnns = parallel_rnn_fit(obs=train_obs, act=train_act,
                                nb_epochs=7500, nb_jobs=25)
        for h in hr:
            print("Horizon: ", h)
            mse, evar = parallel_rnn_test(rnns, test_obs, test_act, int(h))

            k_mse.append(np.min(mse))
            k_evar.append(np.max(evar))
            k.append(h)

    elif args.model == 'lstm':
        # fit lstm
        lstms = parallel_lstm_fit(obs=train_obs, act=train_act,
                                  nb_epochs=150, nb_jobs=25)
        for h in hr:
            print("Horizon: ", h)
            mse, evar = parallel_lstm_test(lstms, test_obs, test_act, int(h))

            k_mse.append(np.min(mse))
            k_evar.append(np.max(evar))
            k.append(h)

    import matplotlib.pyplot as plt
    from tikzplotlib import save

    plt.plot(np.array(k), np.array(k_mse))
    ax = plt.gca()
    ax = beautify(ax)

    save("cart_cartpole_" + str(args.model) + "_mse.tex")
    plt.close()

    plt.plot(np.array(k), np.array(k_evar))
    ax = plt.gca()
    ax = beautify(ax)

    save("cart_cartpole_" + str(args.model) + "_evar.tex")
    plt.close()
