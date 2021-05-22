import numpy as np
import numpy.random as npr

from sds.models import ARHMM, rARHMM
from sds.utils.decorate import ensure_args_are_viable_lists

from joblib import Parallel, delayed

import multiprocessing
nb_cores = multiprocessing.cpu_count()


class Ensemble:

    def __init__(self, nb_states, obs_dim, act_dim=0, nb_lags=1,
                 model_type='rarhmm', ensemble_size=5, **kwargs):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

        self.ensemble_size = ensemble_size

        _type_list = dict(arhmm=ARHMM, rarhmm=rARHMM)
        self.model_type = _type_list[model_type]

        self.models = [self.model_type(self.nb_states, self.obs_dim,
                                       self.act_dim, self.nb_lags, **kwargs)
                       for _ in range(self.ensemble_size)]

    def _parallel_em(self, obs, act, **kwargs):

        def _create_job(model, obs, act,
                        kwargs, seed):

            nb_iter = kwargs.pop('nb_iter', 25)
            prec = kwargs.pop('prec', 1e-4)
            proc_id = seed

            init_mstep_kwargs = kwargs.pop('init_mstep_kwargs', {})
            trans_mstep_kwargs = kwargs.pop('trans_mstep_kwargs', {})
            obs_mstep_kwargs = kwargs.pop('obs_mstep_kwargs', {})

            ll = model.em(obs, act,
                          nb_iter=nb_iter, prec=prec,
                          initialize=True, proc_id=proc_id,
                          init_mstep_kwargs=init_mstep_kwargs,
                          trans_mstep_kwargs=trans_mstep_kwargs,
                          obs_mstep_kwargs=obs_mstep_kwargs)

            return model, ll

        nb_jobs = len(obs)
        kwargs_list = [kwargs.copy() for _ in range(nb_jobs)]
        seeds = np.linspace(0, nb_jobs - 1, nb_jobs, dtype=int)

        results = Parallel(n_jobs=min(nb_jobs, nb_cores), verbose=10, backend='loky')\
            (map(delayed(_create_job), self.models, obs, act, kwargs_list, seeds))

        models, lls = list(map(list, zip(*results)))
        return models, lls

    @ensure_args_are_viable_lists
    def em(self, obs, act=None,
           nb_iter=50, prec=1e-4, initialize=True,
           init_mstep_kwargs={}, trans_mstep_kwargs={},
           obs_mstep_kwargs={}, **kwargs):

        from sds.utils.general import train_validate_split
        train_obs, train_act = train_validate_split(obs, act,
                                                    nb_traj_splits=self.ensemble_size,
                                                    split_trajs=False)[:2]

        self.models, lls = self._parallel_em(train_obs, train_act,
                                             nb_iter=nb_iter, prec=prec,
                                             init_mstep_kwargs=init_mstep_kwargs,
                                             trans_mstep_kwargs=trans_mstep_kwargs,
                                             obs_mstep_kwargs=obs_mstep_kwargs)

        nb_train = []
        nb_total = np.vstack(obs).shape[0]

        train_ll, total_ll = [], []
        for _train_obs, _train_act, _model in zip(train_obs, train_act, self.models):
            nb_train.append(np.vstack(_train_obs).shape[0])
            train_ll.append(_model.log_normalizer(_train_obs, _train_act))
            total_ll.append(_model.log_normalizer(obs, act))

        train_scores = np.hstack(train_ll) / np.hstack(nb_train)
        test_scores = (np.hstack(total_ll) - np.hstack(train_ll))\
                     / (nb_total - np.hstack(nb_train))

        return train_scores, test_scores

    def forcast(self, horizon=None, hist_obs=None, hist_act=None,
                nxt_act=None, stoch=False, average=False, nodes=8):

        nxt_obs = []
        for model in self.models:
            _, _nxt_obs = model.forcast(horizon, hist_obs, hist_act,
                                        nxt_act, stoch, average, nodes)
            nxt_obs.append(np.stack(_nxt_obs, 0))

        nxt_obs = np.stack(nxt_obs, axis=3)
        return np.mean(nxt_obs, axis=3)

    def _kstep_error(self, obs, act, horizon=1, stoch=False, average=False):

        from sklearn.metrics import mean_squared_error, \
            explained_variance_score, r2_score

        hist_obs, hist_act, nxt_act = [], [], []
        forcast, target, prediction = [], [], []

        nb_steps = obs.shape[0] - horizon - self.nb_lags + 1
        for t in range(nb_steps):
            hist_obs.append(obs[:t + self.nb_lags, :])
            hist_act.append(act[:t + self.nb_lags, :])
            nxt_act.append(act[t + self.nb_lags - 1:t + self.nb_lags - 1 + horizon, :])

        hr = [horizon for _ in range(nb_steps)]
        forcast = self.forcast(horizon=hr, hist_obs=hist_obs, hist_act=hist_act,
                               nxt_act=nxt_act, stoch=stoch, average=average)

        for t in range(nb_steps):
            target.append(obs[t + self.nb_lags - 1 + horizon, :])
            prediction.append(forcast[t][-1, :])

        target = np.vstack(target)
        prediction = np.vstack(prediction)

        mse = mean_squared_error(target, prediction)
        smse = 1. - r2_score(target, prediction, multioutput='variance_weighted')
        evar = explained_variance_score(target, prediction, multioutput='variance_weighted')

        return mse, smse, evar

    @ensure_args_are_viable_lists
    def kstep_error(self, obs, act, horizon=1, stoch=False, average=False):

        mse, smse, evar = [], [], []
        for _obs, _act in zip(obs, act):
            _mse, _smse, _evar = self._kstep_error(_obs, _act, horizon, stoch, average)
            mse.append(_mse)
            smse.append(_smse)
            evar.append(_evar)

        return np.mean(mse), np.mean(smse), np.mean(evar)
