import numpy as np
import numpy.random as npr

from sds_numpy import Ensemble


if __name__ == "__main__":

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

    import sds_numpy
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

    ensemble = Ensemble(nb_states=nb_states,
                        type='rarhmm', size=5,
                        dm_obs=dm_obs, dm_act=dm_act,
                        trans_type=trans_type,
                        obs_prior=obs_prior,
                        trans_prior=trans_prior,
                        trans_kwargs=trans_kwargs)

    lls, scores = ensemble.em(train_obs, train_act,
                              nb_iter=3, prec=1e-2,
                              obs_mstep_kwargs=obs_mstep_kwargs,
                              trans_mstep_kwargs=trans_mstep_kwargs)

    print(np.c_[lls, scores])

    hr = [1, 25, 50, 75, 100, 125]
    for h in hr:
        _mse, _smse, _evar = ensemble.kstep_mse(test_obs, test_act, horizon=h)
        print(f"MSE: {_mse}, SMSE:{_smse}, EVAR:{_evar}")
