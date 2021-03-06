import numpy as np
import numpy.random as npr

from sds import rARHMM

import joblib
from joblib import Parallel, delayed

nb_cores = joblib.cpu_count()


def create_job(kwargs):
    # model arguments
    nb_states = kwargs.pop('nb_states')
    trans_type = kwargs.pop('trans_type')
    obs_prior = kwargs.pop('obs_prior')
    trans_prior = kwargs.pop('trans_prior')
    trans_kwargs = kwargs.pop('trans_kwargs')

    # em arguments
    obs = kwargs.pop('obs')
    act = kwargs.pop('act')
    prec = kwargs.pop('prec')
    nb_iter = kwargs.pop('nb_iter')
    obs_mstep_kwargs = kwargs.pop('obs_mstep_kwargs')
    trans_mstep_kwargs = kwargs.pop('trans_mstep_kwargs')

    process_id = kwargs.pop('process_id')

    train_obs, train_act, test_obs, test_act = [], [], [], []
    train_idx = npr.choice(a=len(obs), size=int(0.8 * len(obs)), replace=False)
    for i in range(len(obs)):
        if i in train_idx:
            train_obs.append(obs[i])
            train_act.append(act[i])
        else:
            test_obs.append(obs[i])
            test_act.append(act[i])

    dm_obs = train_obs[0].shape[-1]
    dm_act = train_act[0].shape[-1]

    rarhmm = rARHMM(nb_states, dm_obs, dm_act,
                    trans_type=trans_type,
                    obs_prior=obs_prior,
                    trans_prior=trans_prior,
                    trans_kwargs=trans_kwargs)
    rarhmm.initialize(train_obs, train_act)

    rarhmm.em(train_obs, train_act,
              nb_iter=nb_iter, prec=prec,
              obs_mstep_kwargs=obs_mstep_kwargs,
              trans_mstep_kwargs=trans_mstep_kwargs,
              process_id=process_id)

    nb_train = np.vstack(train_obs).shape[0]
    nb_all = np.vstack(obs).shape[0]

    train_ll = rarhmm.log_norm(train_obs, train_act)
    all_ll = rarhmm.log_norm(obs, act)

    score = (all_ll - train_ll) / (nb_all - nb_train)

    return rarhmm, all_ll, score


def parallel_em(nb_jobs=50, **kwargs):
    kwargs_list = []
    for n in range(nb_jobs):
        kwargs['process_id'] = n
        kwargs_list.append(kwargs.copy())

    results = Parallel(n_jobs=min(nb_jobs, nb_cores), verbose=10, backend='loky')\
        (map(delayed(create_job), kwargs_list))
    rarhmms, lls, scores = list(map(list, zip(*results)))
    return rarhmms, lls, scores


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from hips.plotting.colormaps import gradient_cmap
    import seaborn as sns

    sns.set_style("white")
    sns.set_context("talk")

    color_names = ["windows blue", "red", "amber",
                   "faded green", "dusty purple",
                   "orange", "clay", "pink", "greyish",
                   "mint", "light cyan", "steel blue",
                   "forest green", "pastel purple",
                   "salmon", "dark brown"]

    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors)

    import os
    import random
    import torch

    import sds

    random.seed(1337)
    npr.seed(1337)
    torch.manual_seed(1337)
    torch.set_num_threads(1)

    # load all available data
    files = ['data/walker_1.npz', 'data/walker_2.npz', 'data/walker_3.npz']
    # files = ['data/runner_1.npz', 'data/runner_2.npz', 'data/runner_3.npz']

    state, input = [], []
    for _file in files:
        position, velocity = np.load(_file)['q'], np.load(_file)['dq']
        muscle = np.load(_file)['tau']

        state.append(np.vstack((position, velocity)))
        input.append(muscle)

    obs, act = [], []
    for _obs, _act in zip(state, input):
        for i in range(30):
            obs.append(_obs[:, i * 500: (i + 1) * 500].T)
            act.append(_act[:, i * 500: (i + 1) * 500].T)

    _data = np.hstack((np.vstack(obs), np.vstack(act)))

    train_obs, train_act = obs[:-6], act[:-6]
    test_obs, test_act = obs[-6:], act[-6:]

    nb_states = 9
    dm_obs, dm_act = 60, 14

    obs_prior = {'mu0': 0., 'sigma0': 1e64,
                 'nu0': (dm_obs + 1) + 23, 'psi0': 1e-8 * 23}
    obs_mstep_kwargs = {'use_prior': True}

    trans_type = 'neural'
    trans_prior = {'l2_penalty': 1e-32, 'alpha': 1, 'kappa': 5}
    trans_kwargs = {'hidden_layer_sizes': (128, ),
                    'nonlinearity': 'splus', 'device': 'gpu',
                    'norm': {'mean': np.mean(_data, axis=0),
                             'std': np.std(_data, axis=0)}}
    trans_mstep_kwargs = {'nb_iter': 25, 'batch_size': 4096, 'lr': 5e-4}

    models, lls, scores = parallel_em(nb_jobs=1,
                                      nb_states=nb_states,
                                      obs=train_obs, act=train_act,
                                      trans_type=trans_type,
                                      obs_prior=obs_prior,
                                      trans_prior=trans_prior,
                                      trans_kwargs=trans_kwargs,
                                      obs_mstep_kwargs=obs_mstep_kwargs,
                                      trans_mstep_kwargs=trans_mstep_kwargs,
                                      nb_iter=100, prec=1e-2)
    rarhmm = models[np.argmax(scores)]

    # print("rarhmm, stochastic, " + rarhmm.trans_type)
    # print(np.c_[lls, scores])

    # plt.figure(figsize=(8, 8))
    # _, state = rarhmm.viterbi(train_obs, train_act)
    # _seq = npr.choice(len(train_obs))
    #
    # plt.subplot(211)
    # plt.plot(train_obs[_seq])
    # plt.xlim(0, len(train_obs[_seq]))
    #
    # plt.subplot(212)
    # plt.imshow(state[_seq][None, :], aspect="auto",
    #            cmap=cmap, vmin=0, vmax=len(colors) - 1)
    # plt.xlim(0, len(train_obs[_seq]))
    # plt.ylabel("$z_{\\mathrm{inferred}}$")
    # plt.yticks([])
    #
    # plt.show()

    # idx, buffer, hr = 1, 25, 100
    # z, s = rarhmm.forcast(hist_obs=[train_obs[idx][:buffer, :]],
    #                       hist_act=[train_act[idx][:buffer, :]],
    #                       nxt_act=[train_act[idx][buffer:, :]],
    #                       horizon=[hr], average=True)
    #
    # plt.figure()
    # plt.plot(s[0])

    # plt.figure()
    # plt.plot(z[0])

    # plt.figure()
    # plt.plot(train_obs[idx][buffer:buffer + hr])

    # torch.save(rarhmm, open(rarhmm.trans_type + "_rarhmm_walker.pkl", "wb"))

    hr = np.arange(1, 10)
    for h in hr:
        print("MSE: {0[0]}, SMSE:{0[1]}, EVAR:{0[2]}".
              format(rarhmm.kstep_mse(test_obs, test_act, horizon=h)))
