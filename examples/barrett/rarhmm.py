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
    import pickle

    random.seed(1337)
    npr.seed(1337)
    torch.manual_seed(1337)
    torch.set_num_threads(1)

    # import scipy as sc
    # from scipy import io

    file = pickle.load(open('data/sl_wam_sine_v01_g025.pickle', 'rb'))

    from scipy import signal
    fs = 500
    fc = 25  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = signal.butter(2, w, 'lowpass', output='ba')

    nb_traj = 3

    _obs, _act = [], []
    for n in range(nb_traj):
        _obs.append(signal.filtfilt(b, a, np.hstack((file[n][1], file[n][2])).T).T)
        _act.append(signal.filtfilt(b, a, file[n][4].T).T)

    _data = np.hstack((np.vstack(_obs), np.vstack(_act)))

    from sklearn.decomposition import PCA
    scale = PCA(n_components=21, whiten=True)
    _data = scale.fit_transform(_data)

    train_obs, train_act = [], []
    for j in range(17):
        train_obs.append(_data[j * 2500: (j + 1) * 2500, :14])
        train_act.append(_data[j * 2500: (j + 1) * 2500, 14:])

    test_obs, test_act = [], []
    for j in range(17, 18):
        test_obs.append(_data[j * 2500: (j + 1) * 2500, :14])
        test_act.append(_data[j * 2500: (j + 1) * 2500, 14:])

    nb_states = 13
    dm_obs, dm_act = 14, 7

    obs_prior = {'mu0': 0., 'sigma0': 1e64,
                 'nu0': (dm_obs + 1) + 23, 'psi0': 1e-6 * 23}
    obs_mstep_kwargs = {'use_prior': True}

    trans_type = 'neural'
    trans_prior = {'l2_penalty': 1e-16, 'alpha': 1, 'kappa': 5}
    trans_kwargs = {'hidden_layer_sizes': (128, ),
                    'nonlinearity': 'relu', 'device': 'gpu',
                    'norm': {'mean': np.zeros((dm_obs + dm_act)),
                             'std': np.ones((dm_obs + dm_act))}}
    trans_mstep_kwargs = {'nb_iter': 25, 'batch_size': 2048, 'lr': 5e-4}

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

    # torch.save(rarhmm, open(rarhmm.trans_type + "_rarhmm_barrett.pkl", "wb"))

    hr = [1, 25, 50, 75, 100, 125]  # np.arange(10)
    for h in hr:
        print("MSE: {0[0]}, SMSE:{0[1]}, EVAR:{0[2]}".
              format(rarhmm.kstep_mse(test_obs, test_act, horizon=h)))
