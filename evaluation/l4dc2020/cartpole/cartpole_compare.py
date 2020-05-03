import torch
import numpy as np
import numpy.random as npr

from sds import rARHMM, ARHMM
from sds.utils import sample_env

from reg.gp import DynamicMultiTaskGPRegressor
from reg.nn import DynamicNNRegressor
from reg.nn import DynamicRNNRegressor
from reg.nn import DynamicLSTMRegressor

import random

import joblib
from joblib import Parallel, delayed
nb_cores = joblib.cpu_count()

from matplotlib import rc


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


def parallel_arhmm_fit(obs, act, options, nb_jobs):

    def fit_arhmm_job(args):
        obs, act, options = args

        rarhmm = ARHMM(options['nb_states'],
                       options['dm_obs'], options['dm_act'],
                       obs_prior=options['obs_prior'])

        if options['initialize']:
            rarhmm.initialize(obs, act)

        rarhmm.em(obs=obs, act=act,
                  nb_iter=100, prec=1e-4, verbose=True,
                  obs_mstep_kwargs=options['obs_mstep_kwargs'])

        return rarhmm

    from sklearn.model_selection import ShuffleSplit
    spliter = ShuffleSplit(n_splits=nb_jobs, train_size=0.4)

    obs_lists, act_lists = [], []
    for idx, _ in spliter.split(np.arange(len(obs))):
        obs_lists.append([obs[i] for i in idx])
        act_lists.append([act[i] for i in idx])

    args = [(_obs, _act, options)
            for _obs, _act in zip(obs_lists, act_lists)]

    return Parallel(nb_jobs, verbose=10, backend='loky')\
        (map(delayed(fit_arhmm_job), args))


def parallel_arhmm_test(models, obs, act, horizon):

    def test_arhmm_job(args):
        model, obs, act, horizon = args
        return model.kstep_mse(obs, act, horizon=horizon)

    args = [(model, obs, act, horizon) for model in models]

    res = Parallel(len(models), verbose=10, backend='loky')\
        (map(delayed(test_arhmm_job), args))

    return list(map(list, zip(*res)))


def parallel_rarhmm_fit(obs, act, options, nb_jobs):

    def fit_rarhmm_job(args):
        obs, act, options = args

        rarhmm = rARHMM(options['nb_states'],
                        options['dm_obs'], options['dm_act'],
                        obs_prior=options['obs_prior'],
                        trans_type=options['trans_type'],
                        trans_prior=options['trans_prior'],
                        trans_kwargs=options['trans_kwargs'])

        if options['initialize']:
            rarhmm.initialize(obs, act)

        rarhmm.em(obs=obs, act=act,
                  nb_iter=100, prec=1e-4, verbose=True,
                  obs_mstep_kwargs=options['obs_mstep_kwargs'],
                  trans_mstep_kwargs=options['trans_mstep_kwargs'])

        return rarhmm

    from sklearn.model_selection import ShuffleSplit
    spliter = ShuffleSplit(n_splits=nb_jobs, train_size=0.4)

    obs_lists, act_lists = [], []
    for idx, _ in spliter.split(np.arange(len(obs))):
        obs_lists.append([obs[i] for i in idx])
        act_lists.append([act[i] for i in idx])

    args = [(_obs, _act, options)
            for _obs, _act in zip(obs_lists, act_lists)]

    return Parallel(nb_jobs, verbose=10, backend='loky')\
        (map(delayed(fit_rarhmm_job), args))


def parallel_rarhmm_test(models, obs, act, horizon):

    def test_rarhmm_job(args):
        model, obs, act, horizon = args
        return model.kstep_mse(obs, act, horizon=horizon)

    args = [(model, obs, act, horizon) for model in models]

    res = Parallel(len(models), verbose=10, backend='loky')\
        (map(delayed(test_rarhmm_job), args))

    return list(map(list, zip(*res)))


def parallel_gp_fit(obs, act, incremental, preprocess,
                    nb_iter, nb_jobs, gpu):

    def fit_gp_job(args):
        obs, act, incremental, preprocess, nb_iter, gpu = args

        input = np.vstack([np.hstack((_x[:-1, :], _u[:-1, :]))
                           for _x, _u in zip(obs, act)])
        if incremental:
            target = np.vstack([_x[1:, :] - _x[:-1, :] for _x in obs])
        else:
            target = np.vstack([_x[1:, :] for _x in obs])

        gp = DynamicMultiTaskGPRegressor(input_size=input.shape[-1],
                                         target_size=target.shape[-1],
                                         incremental=incremental,
                                         device='gpu' if gpu else 'cpu')
        gp.fit(target, input, nb_iter, preprocess=preprocess)

        return gp

    from sklearn.model_selection import ShuffleSplit
    spliter = ShuffleSplit(n_splits=nb_jobs, train_size=0.4)

    obs_lists, act_lists = [], []
    for idx, _ in spliter.split(np.arange(len(obs))):
        obs_lists.append([obs[i] for i in idx])
        act_lists.append([act[i] for i in idx])

    args = [(_obs, _act, incremental, preprocess, nb_iter, gpu)
            for _obs, _act in zip(obs_lists, act_lists)]

    nb_threads = 6 if gpu else min(nb_jobs, nb_cores)
    return Parallel(nb_threads, verbose=10, backend='loky')\
        (map(delayed(fit_gp_job), args))


def parallel_gp_test(models, obs, act, horizon, gpu):

    def test_gp_job(args):
        model, obs, act, horizon = args
        return model.kstep_mse(obs, act, horizon)

    args = [(model, obs, act, horizon) for model in models]

    nb_threads = 6 if gpu else len(models)
    res = Parallel(nb_threads, verbose=10, backend='loky')\
        (map(delayed(test_gp_job), args))

    return list(map(list, zip(*res)))


def parallel_fnn_fit(obs, act, size, incremental, preprocess,
                     nb_epochs, nb_jobs, gpu):

    def fit_fnn_job(args):
        obs, act, size, incremental, preprocess, nb_epochs, gpu = args

        input = np.vstack([np.hstack((_x[:-1, :], _u[:-1, :]))
                           for _x, _u in zip(obs, act)])
        if incremental:
            target = np.vstack([_x[1:, :] - _x[:-1, :] for _x in obs])
        else:
            target = np.vstack([_x[1:, :] for _x in obs])

        fnn = DynamicNNRegressor([input.shape[-1], size, size, target.shape[-1]],
                                 nonlin='tanh', incremental=incremental,
                                 device='gpu' if gpu else 'cpu')
        fnn.fit(target, input, nb_epochs, batch_size=32, preprocess=preprocess)

        return fnn

    from sklearn.model_selection import ShuffleSplit
    spliter = ShuffleSplit(n_splits=nb_jobs, train_size=0.4)

    obs_lists, act_lists = [], []
    for idx, _ in spliter.split(np.arange(len(obs))):
        obs_lists.append([obs[i] for i in idx])
        act_lists.append([act[i] for i in idx])

    args = [(_obs, _act, size, incremental, preprocess, nb_epochs, gpu)
            for _obs, _act in zip(obs_lists, act_lists)]

    nb_threads = 6 if gpu else min(nb_jobs, nb_cores)
    return Parallel(nb_threads, verbose=10, backend='loky')\
        (map(delayed(fit_fnn_job), args))


def parallel_fnn_test(models, obs, act, horizon, gpu):

    def test_fnn_job(args):
        model, obs, act, horizon = args
        return model.kstep_mse(obs, act, horizon)

    args = [(model, obs, act, horizon) for model in models]

    nb_threads = 6 if gpu else len(models)
    res = Parallel(nb_threads, verbose=10, backend='loky')\
        (map(delayed(test_fnn_job), args))

    return list(map(list, zip(*res)))


def parallel_rnn_fit(obs, act, size, preprocess, nb_epochs, nb_jobs, gpu):

    def fit_rnn_job(args):
        obs, act, size, preprocess, nb_epochs, gpu = args

        input = np.stack((np.hstack((_obs[:-1, :], _act[:-1, :]))
                          for _obs, _act in zip(obs, act)), axis=0)
        target = np.stack((_obs[1:, :] for _obs in obs), axis=0)

        input_size = input.shape[-1]
        target_size = target.shape[-1]

        rnn = DynamicRNNRegressor(input_size, target_size,
                                  hidden_size=size, nb_layers=2,
                                  nonlinearity='tanh',
                                  device='gpu' if gpu else 'cpu')
        rnn.fit(target, input, nb_epochs, preprocess=preprocess)

        return rnn

    from sklearn.model_selection import ShuffleSplit
    spliter = ShuffleSplit(n_splits=nb_jobs, train_size=0.4)

    obs_lists, act_lists = [], []
    for idx, _ in spliter.split(np.arange(len(obs))):
        obs_lists.append([obs[i] for i in idx])
        act_lists.append([act[i] for i in idx])

    args = [(_obs, _act, size, preprocess, nb_epochs, gpu)
            for _obs, _act in zip(obs_lists, act_lists)]

    nb_threads = 6 if gpu else min(nb_jobs, nb_cores)
    return Parallel(nb_threads, verbose=10, backend='loky')\
        (map(delayed(fit_rnn_job), args))


def parallel_rnn_test(models, obs, act, horizon, gpu):

    def test_rnn_job(args):
        model, obs, act, horizon = args
        return model.kstep_mse(obs, act, horizon)

    args = [(model, obs, act, horizon) for model in models]

    nb_threads = 6 if gpu else len(models)
    res = Parallel(nb_threads, verbose=10, backend='loky')\
        (map(delayed(test_rnn_job), args))

    return list(map(list, zip(*res)))


def parallel_lstm_fit(obs, act, size, preprocess, nb_epochs, nb_jobs, gpu):

    def fit_lstm_job(args):
        obs, act, size, preprocess, nb_epochs, gpu = args

        input = np.stack((np.hstack((_obs[:-1, :], _act[:-1, :]))
                          for _obs, _act in zip(obs, act)), axis=0)
        target = np.stack((_obs[1:, :] for _obs in obs), axis=0)

        input_size = input.shape[-1]
        target_size = target.shape[-1]

        lstm = DynamicLSTMRegressor(input_size, target_size,
                                    hidden_size=size, nb_layers=2,
                                    device='gpu' if gpu else 'cpu')
        lstm.fit(target, input, nb_epochs, lr=0.1, preprocess=preprocess)

        return lstm

    from sklearn.model_selection import ShuffleSplit
    spliter = ShuffleSplit(n_splits=nb_jobs, train_size=0.4)

    obs_lists, act_lists = [], []
    for idx, _ in spliter.split(np.arange(len(obs))):
        obs_lists.append([obs[i] for i in idx])
        act_lists.append([act[i] for i in idx])

    args = [(_obs, _act, size, preprocess, nb_epochs, gpu)
            for _obs, _act in zip(obs_lists, act_lists)]

    nb_threads = 6 if gpu else min(nb_jobs, nb_cores)
    return Parallel(nb_threads, verbose=10, backend='loky')\
        (map(delayed(fit_lstm_job), args))


def parallel_lstm_test(models, obs, act, horizon, gpu):

    def test_lstm_job(args):
        model, obs, act, horizon = args
        return model.kstep_mse(obs, act, horizon)

    args = [(model, obs, act, horizon) for model in models]

    nb_threads = 6 if gpu else len(models)
    res = Parallel(nb_threads, verbose=10, backend='loky')\
        (map(delayed(test_lstm_job), args))

    return list(map(list, zip(*res)))


if __name__ == "__main__":

    import os
    import argparse

    import gym
    import sds

    parser = argparse.ArgumentParser(description='Compare SOTA Models on Cartpole')
    parser.add_argument('--env', help='environment observation', default='cart')
    parser.add_argument('--model', help='representation model', default='rarhmm')
    parser.add_argument('--nb_jobs', help='number of data splits', default=24, type=int)
    parser.add_argument('--incremental', help='approximate delta', action='store_true', default=False)
    parser.add_argument('--preprocess', help='whiten data', action='store_true', default=False)
    parser.add_argument('--nn_size', help='size of NN layer', default=64, type=int)
    parser.add_argument('--nb_states', help='number of linear components', default=7, type=int)
    parser.add_argument('--initialize', help='initialize HMM models', action='store_true', default=True)
    parser.add_argument('--no_init', help='do not initialize HMM models', dest='initialize', action='store_false')
    parser.add_argument('--gpu', help='use gpu', action='store_true', default=False)

    args = parser.parse_args()

    import json
    print(json.dumps(vars(args), indent=4))

    random.seed(1337)
    npr.seed(1337)
    torch.manual_seed(1337)
    torch.set_num_threads(1)

    model_string = str(args.model)

    if args.env == 'cart':
        env = gym.make('Cartpole-ID-v1')
    elif args.env == 'polar':
        env = gym.make('Cartpole-ID-v0')
    else:
        raise NotImplementedError

    env._max_episode_steps = 5000
    env.unwrapped._dt = 0.01
    env.unwrapped._sigma = 1e-4
    env.seed(1337)

    nb_train_rollouts, nb_train_steps = 25, 250
    nb_test_rollouts, nb_test_steps = 5, 100

    hr = [1, 5, 10, 15, 20, 25]

    train_obs, train_act = sample_env(env, nb_train_rollouts, nb_train_steps)
    test_obs, test_act = sample_env(env, nb_test_rollouts, nb_test_steps)

    k, k_mse_avg, k_mse_std = [], [], []
    k_smse_avg, k_smse_std = [], []
    k_evar_avg, k_evar_std = [], []

    if args.model == 'arhmm':
        # fit rarhmm
        obs_prior = {'mu0': 0., 'sigma0': 1e64,
                     'nu0': (env.dm_obs + 1) + 23, 'psi0': 1e-4 * 23}
        obs_mstep_kwargs = {'use_prior': True}

        options = {'dm_obs': env.dm_obs, 'dm_act': env.dm_act, 'nb_states': args.nb_states,
                   'obs_prior': obs_prior, 'obs_mstep_kwargs': obs_mstep_kwargs,
                   'initialize': args.initialize}

        rarhmms = parallel_arhmm_fit(obs=train_obs, act=train_act,
                                     options=options, nb_jobs=args.nb_jobs)

        model_string = model_string + '_' + str(args.nb_states)

        for h in hr:
            print("Horizon: ", h)
            mse, smse, evar = parallel_arhmm_test(rarhmms, test_obs, test_act, int(h))

            k_mse_avg.append(np.mean(mse))
            k_mse_std.append(np.std(mse))

            k_smse_avg.append(np.mean(smse))
            k_smse_std.append(np.std(smse))

            k_evar_avg.append(np.mean(evar))
            k_evar_std.append(np.std(evar))

            k.append(h)

    if args.model == 'rarhmm':
        # fit rarhmm
        obs_prior = {'mu0': 0., 'sigma0': 1e64,
                     'nu0': (env.dm_obs + 1) + 23, 'psi0': 1e-4 * 23}
        obs_mstep_kwargs = {'use_prior': True}

        trans_type = 'neural'
        trans_prior = {'l2_penalty': 1e-32, 'alpha': 1, 'kappa': 1}
        trans_mstep_kwargs = {'nb_iter': 50, 'batch_size': 128, 'lr': 5e-4}

        if args.env == 'cart':
            trans_kwargs = {'hidden_layer_sizes': (24,),
                            'norm': {'mean': np.array([0., 0., 0., 0., 0., 0.]),
                                     'std': np.array([5., 1., 1., 5., 10., 5.])}}
        elif args.env == 'polar':
            trans_kwargs = {'hidden_layer_sizes': (24,),
                            'norm': {'mean': np.array([0., 0., 0., 0., 0.]),
                                     'std': np.array([5., np.pi, 5., 10., 5.])}}
        else:
            raise NotImplementedError

        options = {'dm_obs': env.dm_obs, 'dm_act': env.dm_act, 'nb_states': args.nb_states,
                   'obs_prior': obs_prior, 'obs_mstep_kwargs': obs_mstep_kwargs,
                   'trans_type': trans_type, 'trans_prior': trans_prior,
                   'trans_kwargs': trans_kwargs, 'trans_mstep_kwargs': trans_mstep_kwargs,
                   'initialize': args.initialize}

        rarhmms = parallel_rarhmm_fit(obs=train_obs, act=train_act,
                                      options=options, nb_jobs=args.nb_jobs)

        model_string = model_string + '_' + str(args.nb_states)

        for h in hr:
            print("Horizon: ", h)
            mse, smse, evar = parallel_rarhmm_test(rarhmms, test_obs, test_act, int(h))

            k_mse_avg.append(np.mean(mse))
            k_mse_std.append(np.std(mse))

            k_smse_avg.append(np.mean(smse))
            k_smse_std.append(np.std(smse))

            k_evar_avg.append(np.mean(evar))
            k_evar_std.append(np.std(evar))

            k.append(h)

    elif args.model == 'gp':
        # fit gp
        gps = parallel_gp_fit(obs=train_obs, act=train_act,
                              nb_iter=75, nb_jobs=args.nb_jobs,
                              incremental=args.incremental,
                              preprocess=args.preprocess,
                              gpu=args.gpu)

        for h in hr:
            print("Horizon: ", h)
            mse, smse, evar = parallel_gp_test(gps, test_obs, test_act,
                                               int(h), args.gpu)

            k_mse_avg.append(np.mean(mse))
            k_mse_std.append(np.std(mse))

            k_smse_avg.append(np.mean(smse))
            k_smse_std.append(np.std(smse))

            k_evar_avg.append(np.mean(evar))
            k_evar_std.append(np.std(evar))

            k.append(h)

    elif args.model == 'fnn':
        # fit fnn
        fnns = parallel_fnn_fit(obs=train_obs, act=train_act,
                                size=args.nn_size, incremental=args.incremental,
                                preprocess=args.preprocess, nb_epochs=10000,
                                nb_jobs=args.nb_jobs, gpu=args.gpu)

        model_string = model_string + '_' + str(args.nn_size)

        for h in hr:
            print("Horizon: ", h)
            mse, smse, evar = parallel_fnn_test(fnns, test_obs, test_act,
                                                int(h), args.gpu)

            k_mse_avg.append(np.mean(mse))
            k_mse_std.append(np.std(mse))

            k_smse_avg.append(np.mean(smse))
            k_smse_std.append(np.std(smse))

            k_evar_avg.append(np.mean(evar))
            k_evar_std.append(np.std(evar))

            k.append(h)

    elif args.model == 'rnn':
        # fit rnn
        rnns = parallel_rnn_fit(obs=train_obs, act=train_act,
                                size=args.nn_size, preprocess=args.preprocess,
                                nb_epochs=10000, nb_jobs=args.nb_jobs,
                                gpu=args.gpu)

        model_string = model_string + '_' + str(args.nn_size)

        for h in hr:
            print("Horizon: ", h)
            mse, smse, evar = parallel_rnn_test(rnns, test_obs, test_act,
                                                int(h), args.gpu)

            k_mse_avg.append(np.mean(mse))
            k_mse_std.append(np.std(mse))

            k_smse_avg.append(np.mean(smse))
            k_smse_std.append(np.std(smse))

            k_evar_avg.append(np.mean(evar))
            k_evar_std.append(np.std(evar))

            k.append(h)

    elif args.model == 'lstm':
        # fit lstm
        lstms = parallel_lstm_fit(obs=train_obs, act=train_act,
                                  size=args.nn_size, preprocess=args.preprocess,
                                  nb_epochs=150, nb_jobs=args.nb_jobs,
                                  gpu=args.gpu)

        model_string = model_string + '_' + str(args.nn_size)

        for h in hr:
            print("Horizon: ", h)
            mse, smse, evar = parallel_lstm_test(lstms, test_obs, test_act,
                                                 int(h), args.gpu)

            k_mse_avg.append(np.mean(mse))
            k_mse_std.append(np.std(mse))

            k_smse_avg.append(np.mean(smse))
            k_smse_std.append(np.std(smse))

            k_evar_avg.append(np.mean(evar))
            k_evar_std.append(np.std(evar))

            k.append(h)

    import matplotlib.pyplot as plt
    from tikzplotlib import save

    plt.figure()
    plt.errorbar(np.array(k), np.array(k_mse_avg),
                 yerr=np.array(k_mse_std),
                 fmt='-o', capsize=7, markersize=5)
    ax = plt.gca()
    ax = beautify(ax)

    save(str(args.env) + "_cartpole_" + str(model_string) + "_mse.tex")
    plt.close()

    plt.figure()
    plt.errorbar(np.array(k), np.array(k_smse_avg),
                 yerr=np.array(k_smse_std),
                 fmt='-o', capsize=7, markersize=5)
    ax = plt.gca()
    ax = beautify(ax)

    save(str(args.env) + "_cartpole_" + str(model_string) + "_smse.tex")
    plt.close()

    plt.figure()
    plt.errorbar(np.array(k), np.array(k_evar_avg),
                 yerr=np.array(k_evar_std),
                 fmt='-o', capsize=7, markersize=5)
    ax = plt.gca()
    ax = beautify(ax)

    save(str(args.env) + "_cartpole_" + str(model_string) + "_evar.tex")
    plt.close()
