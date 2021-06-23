import numpy as np
import numpy.random as npr

from sds.models import AutoRegressiveHiddenMarkovModel
from sds.models import RecurrentAutoRegressiveHiddenMarkovModel
from sds.models import ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel
from sds.models import AutoRegressiveClosedLoopHiddenMarkovModel
from sds.models import HybridController

from sds.utils.decorate import ensure_args_are_viable

from joblib import Parallel, delayed

import multiprocessing
nb_cores = multiprocessing.cpu_count()


class EnsembleHiddenMarkovModel:

    def __init__(self, nb_states, obs_dim, act_dim=0, obs_lag=1,
                 model_type='rarhmm', ensemble_size=5, **kwargs):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_lag = obs_lag

        self.ensemble_size = ensemble_size

        type_list = dict(arhmm=AutoRegressiveHiddenMarkovModel,
                         rarhmm=RecurrentAutoRegressiveHiddenMarkovModel)
        self.model_type = type_list[model_type]

        self.models = [self.model_type(self.nb_states, self.obs_dim,
                                       self.act_dim, self.obs_lag, **kwargs)
                       for _ in range(self.ensemble_size)]

    def _parallel_em(self, obs, act, **kwargs):

        def _create_job(model, obs, act,
                        kwargs, seed):

            nb_iter = kwargs.get('nb_iter', 25)
            prec = kwargs.get('prec', 1e-4)
            initialize = kwargs.get('initialize', True)
            proc_id = seed

            init_state_mstep_kwargs = kwargs.get('init_state_mstep_kwargs', {})
            init_obs_mstep_kwargs = kwargs.get('init_obs_mstep_kwargs', {})
            trans_mstep_kwargs = kwargs.get('trans_mstep_kwargs', {})
            obs_mstep_kwargs = kwargs.get('obs_mstep_kwargs', {})

            ll = model.em(obs, act,
                          nb_iter=nb_iter, prec=prec,
                          initialize=initialize, proc_id=proc_id,
                          init_state_mstep_kwargs=init_state_mstep_kwargs,
                          init_obs_mstep_kwargs=init_obs_mstep_kwargs,
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

    @ensure_args_are_viable
    def em(self, obs, act=None,
           nb_iter=50, prec=1e-4, initialize=True,
           init_state_mstep_kwargs={},
           init_obs_mstep_kwargs={},
           trans_mstep_kwargs={},
           obs_mstep_kwargs={}, **kwargs):

        from sds.utils.general import train_test_split
        train_obs, train_act = train_test_split(obs, act,
                                                nb_traj_splits=self.ensemble_size,
                                                split_trajs=False)[:2]

        self.models, lls = self._parallel_em(train_obs, train_act,
                                             nb_iter=nb_iter, prec=prec, initialize=initialize,
                                             init_state_mstep_kwargs=init_state_mstep_kwargs,
                                             init_obs_mstep_kwargs=init_obs_mstep_kwargs,
                                             trans_mstep_kwargs=trans_mstep_kwargs,
                                             obs_mstep_kwargs=obs_mstep_kwargs)

        nb_train = [np.vstack(x).shape[0] for x in train_obs]
        nb_total = np.vstack(obs).shape[0]

        train_ll, total_ll = [], []
        for x, u, m in zip(train_obs, train_act, self.models):
            train_ll.append(m.log_normalizer(x, u))
            total_ll.append(m.log_normalizer(obs, act))

        train_scores = np.hstack(train_ll) / np.hstack(nb_train)
        test_scores = (np.hstack(total_ll) - np.hstack(train_ll))\
                     / (nb_total - np.hstack(nb_train))

        return train_scores, test_scores

    def step(self, hist_obs, hist_act, stoch=False, average=False):
        nxt_obs = np.zeros((self.ensemble_size, self.obs_dim))
        for i, model in enumerate(self.models):
            _, nxt_obs[i] = model.step(hist_obs, hist_act, stoch, average)
        return np.mean(nxt_obs, axis=0)

    def forcast(self, horizon=1, hist_obs=None, hist_act=None,
                nxt_act=None, stoch=False, average=False):
        if isinstance(horizon, int) and isinstance(hist_obs, np.ndarray):
            nxt_obs = []
            for m in self.models:
                nxt_obs.append(m.forcast(horizon, hist_obs, hist_act, nxt_act, stoch, average)[1])
            return np.mean(np.stack(nxt_obs, axis=0), axis=0)
        else:
            nxt_obs = []
            for m in self.models:
                _nxt_obs = m.forcast(horizon, hist_obs, hist_act, nxt_act, stoch, average)[1]
                nxt_obs.append(np.stack(_nxt_obs, 0))
            return np.mean(np.stack(nxt_obs, axis=0), axis=0)

    def _kstep_error(self, obs, act, horizon=1, stoch=False, average=False):

        from sklearn.metrics import mean_squared_error, \
            explained_variance_score, r2_score

        hist_obs, hist_act, nxt_act = [], [], []
        forcast, target, prediction = [], [], []

        nb_steps = obs.shape[0] - horizon - self.obs_lag + 1
        for t in range(nb_steps):
            hist_obs.append(obs[:t + self.obs_lag, :])
            hist_act.append(act[:t + self.obs_lag, :])
            nxt_act.append(act[t + self.obs_lag - 1:t + self.obs_lag - 1 + horizon, :])

        hr = [horizon for _ in range(nb_steps)]
        forcast = self.forcast(horizon=hr, hist_obs=hist_obs, hist_act=hist_act,
                               nxt_act=nxt_act, stoch=stoch, average=average)

        for t in range(nb_steps):
            target.append(obs[t + self.obs_lag - 1 + horizon, :])
            prediction.append(forcast[t][-1, :])

        target = np.vstack(target)
        prediction = np.vstack(prediction)

        mse = mean_squared_error(target, prediction)
        smse = 1. - r2_score(target, prediction, multioutput='variance_weighted')
        evar = explained_variance_score(target, prediction, multioutput='variance_weighted')

        return mse, smse, evar

    @ensure_args_are_viable
    def kstep_error(self, obs, act, horizon=1, stoch=False, average=False):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            return self._kstep_error(obs, act, horizon, stoch, average)
        else:
            def inner(obs, act):
                return self.kstep_error.__wrapped__(self, obs, act, horizon, stoch, average)
            res = list(map(inner, obs, act))
            mse, smse, evar = list(map(list, zip(*res)))
            return np.mean(mse), np.mean(smse), np.mean(evar)


class EnsembleClosedLoopHiddenMarkovModel:

    def __init__(self, nb_states, obs_dim, act_dim, obs_lag=1,
                 ensemble_size=6, **kwargs):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_lag = obs_lag

        self.ensemble_size = ensemble_size

        self.models = [ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel(self.nb_states, self.obs_dim,
                                                                          self.act_dim, self.obs_lag, **kwargs)
                       for _ in range(self.ensemble_size)]

    def _parallel_em(self, obs, act, **kwargs):

        def _create_job(model, obs, act,
                        kwargs, seed):

            nb_iter = kwargs.get('nb_iter', 25)
            prec = kwargs.get('prec', 1e-4)
            initialize = kwargs.get('initialize', True)
            proc_id = seed

            init_state_mstep_kwargs = kwargs.get('init_state_mstep_kwargs', {})
            init_obs_mstep_kwargs = kwargs.get('init_obs_mstep_kwargs', {})
            trans_mstep_kwargs = kwargs.get('trans_mstep_kwargs', {})
            obs_mstep_kwargs = kwargs.get('obs_mstep_kwargs', {})
            ctl_mstep_kwargs = kwargs.get('ctl_mstep_kwargs', {})

            ll = model.em(obs, act,
                          nb_iter=nb_iter, prec=prec,
                          initialize=initialize, proc_id=proc_id,
                          init_state_mstep_kwargs=init_state_mstep_kwargs,
                          init_obs_mstep_kwargs=init_obs_mstep_kwargs,
                          trans_mstep_kwargs=trans_mstep_kwargs,
                          obs_mstep_kwargs=obs_mstep_kwargs,
                          ctl_mstep_kwargs=ctl_mstep_kwargs)

            return model, ll

        nb_jobs = len(obs)
        kwargs_list = [kwargs.copy() for _ in range(nb_jobs)]
        seeds = np.linspace(0, nb_jobs - 1, nb_jobs, dtype=int)

        results = Parallel(n_jobs=min(nb_jobs, nb_cores), verbose=10, backend='loky')\
            (map(delayed(_create_job), self.models, obs, act, kwargs_list, seeds))

        models, lls = list(map(list, zip(*results)))
        return models, lls

    @ensure_args_are_viable
    def em(self, obs, act=None,
           nb_iter=50, prec=1e-4, initialize=True,
           init_state_mstep_kwargs={},
           init_obs_mstep_kwargs={},
           trans_mstep_kwargs={},
           obs_mstep_kwargs={},
           ctl_mstep_kwargs={}, **kwargs):

        from sds.utils.general import train_test_split
        train_obs, train_act = train_test_split(obs, act,
                                                nb_traj_splits=self.ensemble_size,
                                                split_trajs=False)[:2]

        self.models, lls = self._parallel_em(train_obs, train_act,
                                             nb_iter=nb_iter, prec=prec, initialize=initialize,
                                             init_state_mstep_kwargs=init_state_mstep_kwargs,
                                             init_obs_mstep_kwargs=init_obs_mstep_kwargs,
                                             trans_mstep_kwargs=trans_mstep_kwargs,
                                             obs_mstep_kwargs=obs_mstep_kwargs,
                                             ctl_mstep_kwargs=ctl_mstep_kwargs)

        nb_train = [np.vstack(x).shape[0] for x in train_obs]
        nb_total = np.vstack(obs).shape[0]

        train_ll, total_ll = [], []
        for x, u, m in zip(train_obs, train_act, self.models):
            train_ll.append(m.log_normalizer(x, u))
            total_ll.append(m.log_normalizer(obs, act))

        train_scores = np.hstack(train_ll) / np.hstack(nb_train)
        test_scores = (np.hstack(total_ll) - np.hstack(train_ll))\
                     / (nb_total - np.hstack(nb_train))

        return train_scores, test_scores


class EnsembleAutoRegressiveClosedLoopHiddenMarkovModel:

    def __init__(self, nb_states, obs_dim, act_dim, obs_lag=1,
                 ctl_lag=1, ensemble_size=6, **kwargs):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_lag = obs_lag
        self.ctl_lag = ctl_lag

        self.ensemble_size = ensemble_size

        self.models = [AutoRegressiveClosedLoopHiddenMarkovModel(self.nb_states, self.obs_dim, self.act_dim,
                                                                 self.obs_lag, self.ctl_lag, **kwargs)
                       for _ in range(self.ensemble_size)]

    def _parallel_em(self, obs, act, **kwargs):

        def _create_job(model, obs, act,
                        kwargs, seed):

            nb_iter = kwargs.get('nb_iter', 25)
            prec = kwargs.get('prec', 1e-4)
            initialize = kwargs.get('initialize', True)
            proc_id = seed

            init_state_mstep_kwargs = kwargs.get('init_state_mstep_kwargs', {})
            init_obs_mstep_kwargs = kwargs.get('init_obs_mstep_kwargs', {})
            init_ctl_mstep_kwargs = kwargs.get('init_ctl_mstep_kwargs', {})
            trans_mstep_kwargs = kwargs.get('trans_mstep_kwargs', {})
            obs_mstep_kwargs = kwargs.get('obs_mstep_kwargs', {})
            ctl_mstep_kwargs = kwargs.get('ctl_mstep_kwargs', {})

            ll = model.em(obs, act,
                          nb_iter=nb_iter, prec=prec,
                          initialize=initialize, proc_id=proc_id,
                          init_state_mstep_kwargs=init_state_mstep_kwargs,
                          init_obs_mstep_kwargs=init_obs_mstep_kwargs,
                          init_ctl_mstep_kwargs=init_ctl_mstep_kwargs,
                          trans_mstep_kwargs=trans_mstep_kwargs,
                          obs_mstep_kwargs=obs_mstep_kwargs,
                          ctl_mstep_kwargs=ctl_mstep_kwargs)

            return model, ll

        nb_jobs = len(obs)
        kwargs_list = [kwargs.copy() for _ in range(nb_jobs)]
        seeds = np.linspace(0, nb_jobs - 1, nb_jobs, dtype=int)

        results = Parallel(n_jobs=min(nb_jobs, nb_cores), verbose=10, backend='loky')\
            (map(delayed(_create_job), self.models, obs, act, kwargs_list, seeds))

        models, lls = list(map(list, zip(*results)))
        return models, lls

    @ensure_args_are_viable
    def em(self, obs, act=None,
           nb_iter=50, prec=1e-4, initialize=True,
           init_state_mstep_kwargs={},
           init_obs_mstep_kwargs={},
           init_ctl_mstep_kwargs={},
           trans_mstep_kwargs={},
           obs_mstep_kwargs={},
           ctl_mstep_kwargs={}, **kwargs):

        from sds.utils.general import train_test_split
        train_obs, train_act = train_test_split(obs, act,
                                                nb_traj_splits=self.ensemble_size,
                                                split_trajs=False)[:2]

        self.models, lls = self._parallel_em(train_obs, train_act,
                                             nb_iter=nb_iter, prec=prec, initialize=initialize,
                                             init_state_mstep_kwargs=init_state_mstep_kwargs,
                                             init_obs_mstep_kwargs=init_obs_mstep_kwargs,
                                             init_ctl_mstep_kwargs=init_ctl_mstep_kwargs,
                                             trans_mstep_kwargs=trans_mstep_kwargs,
                                             obs_mstep_kwargs=obs_mstep_kwargs,
                                             ctl_mstep_kwargs=ctl_mstep_kwargs)

        nb_train = [np.vstack(x).shape[0] for x in train_obs]
        nb_total = np.vstack(obs).shape[0]

        train_ll, total_ll = [], []
        for x, u, m in zip(train_obs, train_act, self.models):
            train_ll.append(m.log_normalizer(x, u))
            total_ll.append(m.log_normalizer(obs, act))

        train_scores = np.hstack(train_ll) / np.hstack(nb_train)
        test_scores = (np.hstack(total_ll) - np.hstack(train_ll))\
                     / (nb_total - np.hstack(nb_train))

        return train_scores, test_scores


class EnsembleHybridController:

    def __init__(self, dynamics, ensemble_size=6, **kwargs):

        self.dynamics = dynamics
        self.ensemble_size = ensemble_size

        self.models = [HybridController(dynamics, **kwargs)
                       for _ in range(self.ensemble_size)]

    def _parallel_em(self, obs, act, **kwargs):

        def _create_job(model, obs, act,
                        kwargs, seed):

            nb_iter = kwargs.get('nb_iter', 25)
            prec = kwargs.get('prec', 1e-4)
            initialize = kwargs.get('initialize', True)
            proc_id = seed

            ctl_mstep_kwargs = kwargs.get('ctl_mstep_kwargs', {})

            ll = model.em(obs, act,
                          nb_iter=nb_iter, prec=prec,
                          initialize=initialize, proc_id=proc_id,
                          ctl_mstep_kwargs=ctl_mstep_kwargs)

            return model, ll

        nb_jobs = len(obs)
        kwargs_list = [kwargs.copy() for _ in range(nb_jobs)]
        seeds = np.linspace(0, nb_jobs - 1, nb_jobs, dtype=int)

        results = Parallel(n_jobs=min(nb_jobs, nb_cores), verbose=10, backend='loky')\
            (map(delayed(_create_job), self.models, obs, act, kwargs_list, seeds))

        models, lls = list(map(list, zip(*results)))
        return models, lls

    @ensure_args_are_viable
    def em(self, obs, act=None,
           nb_iter=50, prec=1e-4, initialize=True,
           ctl_mstep_kwargs={}, **kwargs):

        from sds.utils.general import train_test_split
        train_obs, train_act = train_test_split(obs, act,
                                                nb_traj_splits=self.ensemble_size,
                                                split_trajs=False)[:2]

        self.models, lls = self._parallel_em(train_obs, train_act,
                                             nb_iter=nb_iter, prec=prec,
                                             initialize=initialize,
                                             ctl_mstep_kwargs=ctl_mstep_kwargs)

        nb_train = [np.vstack(x).shape[0] for x in train_obs]
        nb_total = np.vstack(obs).shape[0]

        train_ll, total_ll = [], []
        for x, u, m in zip(train_obs, train_act, self.models):
            train_ll.append(m.log_normalizer(x, u))
            total_ll.append(m.log_normalizer(obs, act))

        train_scores = np.hstack(train_ll) / np.hstack(nb_train)
        test_scores = (np.hstack(total_ll) - np.hstack(train_ll))\
                     / (nb_total - np.hstack(nb_train))

        return train_scores, test_scores
