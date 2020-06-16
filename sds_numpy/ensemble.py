import numpy as np
import numpy.random as npr

from sds_numpy import HMM, ARHMM, rARHMM
from sds_numpy.utils import ensure_args_are_viable_lists

from joblib import Parallel, delayed

import multiprocessing
nb_cores = multiprocessing.cpu_count()


class Ensemble():

    def __init__(self, nb_states, dm_obs, dm_act=0,
                 type='rarhmm', size=5, **kwargs):

        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.size = size

        _list = dict(hmm=HMM, arhmm=ARHMM, rarhmm=rARHMM)
        self.type = _list[type]

        self.models = [self.type(self.nb_states, self.dm_obs,
                                 self.dm_act, **kwargs)
                       for _ in range(self.size)]

    def _parallel_em(self, obs, act, **kwargs):

        def _job(kwargs):
            obs = kwargs.pop('obs')
            act = kwargs.pop('act')

            model = kwargs.pop('model')

            prec = kwargs.pop('prec', 1e-2)
            nb_iter = kwargs.pop('nb_iter', 1e-2)
            obs_mstep_kwargs = kwargs.pop('obs_mstep_kwargs', {})
            trans_mstep_kwargs = kwargs.pop('trans_mstep_kwargs', {})

            # model.initialize(obs, act)
            ll = model.em(obs, act,
                          nb_iter=nb_iter, prec=prec,
                          obs_mstep_kwargs=obs_mstep_kwargs,
                          trans_mstep_kwargs=trans_mstep_kwargs)

            return model, ll

        nb_jobs = len(obs)

        kwargs_list = []
        for n in range(nb_jobs):
            kwargs['obs'] = obs[n]
            kwargs['act'] = act[n]
            kwargs['model'] = self.models[n]
            kwargs_list.append(kwargs.copy())

        results = Parallel(n_jobs=min(nb_jobs, nb_cores), verbose=10, backend='loky')\
            (map(delayed(_job), kwargs_list))

        models, lls = list(map(list, zip(*results)))
        return models, lls

    @ensure_args_are_viable_lists
    def em(self, obs, act=None, nb_iter=50, prec=1e-4,
           init_mstep_kwargs={}, trans_mstep_kwargs={},
           obs_mstep_kwargs={}):

        train_obs, train_act = [], []
        for n in range(self.size):
            _train_obs, _train_act = [], []
            idx = npr.choice(a=len(obs), size=int(0.8 * len(obs)), replace=False)
            for i in range(len(obs)):
                if i in idx:
                    _train_obs.append(obs[i])
                    _train_act.append(act[i])

            train_obs.append(_train_obs)
            train_act.append(_train_act)

        self.models, lls = self._parallel_em(train_obs, train_act,
                                             nb_iter=nb_iter, prec=prec,
                                             init_mstep_kwargs=init_mstep_kwargs,
                                             trans_mstep_kwargs=trans_mstep_kwargs,
                                             obs_mstep_kwargs=obs_mstep_kwargs)

        nb_train = []
        nb_total = np.vstack(obs).shape[0]

        train_ll, total_all = [], []
        for _train_obs, _train_act, _model in zip(train_obs, train_act, self.models):
            nb_train.append(np.vstack(_train_obs).shape[0])
            train_ll.append(_model.log_norm(_train_obs, _train_act))
            total_all.append(_model.log_norm(obs, act))

        scores = (np.hstack(total_all) - np.hstack(train_ll))\
                 / (nb_total - np.hstack(nb_train))

        return total_all, scores

    def forcast(self, hist_obs=None, hist_act=None, nxt_act=None,
                horizon=None, stoch=False, average=False):

        nxt_state, nxt_obs = [], []
        for model in self.models:
            _, _nxt_obs = model.forcast(hist_obs, hist_act, nxt_act,
                                        horizon, stoch, average)
            nxt_obs.append(np.stack(_nxt_obs, 0))

        nxt_obs = np.stack(nxt_obs, axis=3)
        return np.mean(nxt_obs, axis=3)

    @ensure_args_are_viable_lists
    def kstep_mse(self, obs, act, horizon=1, stoch=False, average=False):

        from sklearn.metrics import mean_squared_error,\
            explained_variance_score, r2_score

        mse, smse, evar = [], [], []
        for _obs, _act in zip(obs, act):
            _hist_obs, _hist_act, _nxt_act = [], [], []
            _target, _prediction = [], []

            _nb_steps = _obs.shape[0] - horizon
            for t in range(_nb_steps):
                _hist_obs.append(_obs[:t + 1, :])
                _hist_act.append(_act[:t + 1, :])
                _nxt_act.append(_act[t: t + horizon, :])

            _hr = [horizon for _ in range(_nb_steps)]
            _forcast = self.forcast(hist_obs=_hist_obs, hist_act=_hist_act,
                                    nxt_act=_nxt_act, horizon=_hr, stoch=stoch,
                                    average=average)

            for t in range(_nb_steps):
                _target.append(_obs[t + horizon, :])
                _prediction.append(_forcast[t][-1, :])

            _target = np.vstack(_target)
            _prediction = np.vstack(_prediction)

            _mse = mean_squared_error(_target, _prediction)
            _smse = 1. - r2_score(_target, _prediction, multioutput='variance_weighted')
            _evar = explained_variance_score(_target, _prediction, multioutput='variance_weighted')

            mse.append(_mse)
            smse.append(_smse)
            evar.append(_evar)

        return np.mean(mse), np.mean(smse), np.mean(evar)
