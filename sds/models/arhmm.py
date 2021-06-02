import numpy as np
import numpy.random as npr

from scipy.special import logsumexp

from sds.models import HiddenMarkovModel

from sds.initial import InitGaussianObservation
from sds.initial import BayesianInitGaussianObservation
from sds.initial import BayesianInitDiagonalGaussianObservation

from sds.observations import AutoRegressiveGaussianObservation
from sds.observations import BayesianAutoRegressiveGaussianObservation
from sds.observations import BayesianAutoRegressiveDiagonalGaussianObservation
from sds.observations import BayesianAutoRegressiveTiedGaussianObservation
from sds.observations import BayesianAutoRegressiveTiedDiagonalGaussianObservation

from sds.utils.decorate import ensure_args_are_viable

from pathos.multiprocessing import ProcessPool


class AutoRegressiveHiddenMarkovModel(HiddenMarkovModel):

    def __init__(self, nb_states, obs_dim, act_dim=0, obs_lag=1,
                 algo_type='ML', init_obs_type='full', obs_type='full',
                 init_state_prior=None, init_obs_prior=None, trans_prior=None, obs_prior=None,
                 init_state_kwargs={}, init_obs_kwargs={}, trans_kwargs={}, obs_kwargs={}):

        super(AutoRegressiveHiddenMarkovModel, self).__init__(nb_states, obs_dim, act_dim,
                                                              init_state_kwargs=init_state_kwargs,
                                                              trans_kwargs=trans_kwargs)

        self.obs_lag = obs_lag

        self.algo_type = algo_type
        self.init_obs_type = init_obs_type
        self.obs_type = obs_type

        self.init_state_prior = init_state_prior
        self.init_obs_prior = init_obs_prior
        self.trans_prior = trans_prior
        self.obs_prior = obs_prior

        self.init_state_kwargs = init_state_kwargs
        self.init_obs_kwargs = init_obs_kwargs
        self.trans_kwargs = trans_kwargs
        self.obs_kwargs = obs_kwargs

        if algo_type == 'ML':
            self.init_observation = InitGaussianObservation(self.nb_states, self.obs_dim,
                                                            self.act_dim, self.obs_lag, **init_obs_kwargs)
            self.observations = AutoRegressiveGaussianObservation(self.nb_states, self.obs_dim,
                                                                  self.act_dim, self.obs_lag, **obs_kwargs)
        else:
            if init_obs_type == 'full':
                self.init_observation = BayesianInitGaussianObservation(self.nb_states, self.obs_dim, self.act_dim,
                                                                        self.obs_lag, prior=init_obs_prior,
                                                                        **init_obs_kwargs)
            elif init_obs_type == 'diagonal':
                self.init_observation = BayesianInitDiagonalGaussianObservation(self.nb_states, self.obs_dim, self.act_dim,
                                                                                self.obs_lag, prior=init_obs_prior,
                                                                                **init_obs_kwargs)
            else:
                raise NotImplementedError

            if obs_type == 'full':
                self.observations = BayesianAutoRegressiveGaussianObservation(self.nb_states, self.obs_dim, self.act_dim,
                                                                              self.obs_lag, prior=obs_prior, **obs_kwargs)
            elif obs_type == 'diagonal':
                self.observations = BayesianAutoRegressiveDiagonalGaussianObservation(self.nb_states, self.obs_dim, self.act_dim,
                                                                                      self.obs_lag, prior=obs_prior, **obs_kwargs)
            elif obs_type == 'tied-full':
                self.observations = BayesianAutoRegressiveTiedGaussianObservation(self.nb_states, self.obs_dim, self.act_dim,
                                                                                  self.obs_lag, prior=obs_prior, **obs_kwargs)
            elif obs_type == 'tied-diagonal':
                self.observations = BayesianAutoRegressiveTiedDiagonalGaussianObservation(self.nb_states, self.obs_dim, self.act_dim,
                                                                                          self.obs_lag, prior=obs_prior, **obs_kwargs)
            else:
                raise NotImplementedError

    @property
    def params(self):
        return self.init_state.params,\
               self.init_observation.params,\
               self.transitions.params,\
               self.observations.params

    @params.setter
    def params(self, value):
        self.init_state.params = value[0]
        self.init_observation.params = value[1]
        self.transitions.params = value[2]
        self.observations.params = value[3]

    def permute(self, perm):
        self.init_state.permute(perm)
        self.init_observation.permute(perm)
        self.transitions.permute(perm)
        self.observations.permute(perm)

    @ensure_args_are_viable
    def initialize(self, obs, act=None, **kwargs):
        super(AutoRegressiveHiddenMarkovModel, self).initialize(obs, act)
        self.init_observation.initialize(obs)

    @ensure_args_are_viable
    def log_likelihoods(self, obs, act=None):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            loginit, logtrans, arlogobs =\
                super(AutoRegressiveHiddenMarkovModel, self).log_likelihoods(obs, act)

            ilogobs = self.init_observation.log_likelihood(obs)
            logobs = np.vstack((ilogobs, arlogobs))
            return loginit, logtrans, logobs
        else:
            def inner(obs, act):
                return self.log_likelihoods.__wrapped__(self, obs, act)
            result = map(inner, obs, act)
            return list(map(list, zip(*result)))

    def mstep(self, gamma, zeta,
              obs, act,
              init_state_mstep_kwargs,
              trans_mstep_kwargs,
              obs_mstep_kwargs, **kwargs):

        super(AutoRegressiveHiddenMarkovModel, self).mstep(gamma, zeta, obs, act,
                                                           init_state_mstep_kwargs,
                                                           trans_mstep_kwargs,
                                                           obs_mstep_kwargs)
        init_obs_mstep_kwargs = kwargs.get('init_obs_mstep_kwargs', {})
        self.init_observation.mstep(gamma, obs, **init_obs_mstep_kwargs)

    @ensure_args_are_viable
    def smoothed_observation(self, obs, act=None):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            loglikhds = self.log_likelihoods(obs, act)
            alpha, norm = self.forward(*loglikhds)
            beta = self.backward(*loglikhds, scale=norm)
            gamma = self.posterior(alpha, beta)

            iobs = self.init_observation.smooth(gamma, obs)
            arobs = self.observations.smooth(gamma, obs, act)
            return np.vstack((iobs, arobs))
        else:
            def inner(obs, act):
                return self.smoothed_observation.__wrapped__(self, obs, act)
            return list(map(inner, obs, act))

    def step(self, hist_obs, hist_act, stoch=False, average=False):
        # take last belief state from filter
        belief = self.filtered_state(hist_obs, hist_act)[-1]

        nxt_state = np.zeros((1, ), dtype=np.int64)
        nxt_obs = np.zeros((self.obs_dim, ))
        if average:
            # empty discrete state when averaging
            nxt_state = None

            # transition given last obs and act
            xl, ul = hist_obs[-1], hist_act[-1]
            trans = self.transitions.matrix(xl, ul)

            # update belief with transition
            belief = trans.T @ belief
            belief /= belief.sum()

            # average observations under belief
            for k in range(self.nb_states):
                xr, ul = hist_obs[-self.obs_lag:], hist_act[-1]
                nxt_obs += belief[k] * (self.observations.sample(k, xr, ul, ar=True) if stoch
                                        else self.observations.mean(k, xr, ul, ar=True))
        else:
            # sample last discrete state
            zl = npr.choice(self.nb_states, p=belief) if stoch\
                 else np.argmax(belief)

            # sample next discrete state
            xl, ul = hist_obs[-1], hist_act[-1]
            nxt_state = self.transitions.sample(zl, xl, ul) if stoch \
                        else self.transitions.likeliest(zl, xl, ul)

            # sample next observation
            zn = nxt_state
            xr = hist_obs[-self.obs_lag:]
            nxt_obs = self.observations.sample(zn, xr, ul, ar=True) if stoch\
                      else self.observations.mean(zn, xr, ul, ar=True)

        return nxt_state, nxt_obs

    def _sample(self, horizon, act=None, seed=None):
        npr.seed(seed)

        act = np.zeros((horizon, self.act_dim)) if act is None else act
        obs = np.zeros((horizon, self.obs_dim))
        state = np.zeros((horizon,), np.int64)

        state[0] = self.init_state.sample()
        obs[0] = self.init_observation.sample(state[0])

        for t in range(1, self.obs_lag):
            state[t] = self.transitions.sample(state[t - 1], obs[t - 1], act[t - 1])
            obs[t] = self.init_observation.sample(state[t])

        for t in range(self.obs_lag, horizon):
            state[t] = self.transitions.sample(state[t - 1], obs[t - 1], act[t - 1])
            obs[t] = self.observations.sample(state[t], obs[t - self.obs_lag:t], act[t - 1], ar=True)

        return state, obs

    def sample(self, horizon, act=None, nodes=8):
        act = [None] * len(horizon) if act is None else act
        seeds = [i for i in range(len(horizon))]

        pool = ProcessPool(nodes=nodes)
        res = pool.map(self._sample, horizon, act, seeds)
        pool.clear()

        state, obs = list(map(list, zip(*res)))
        return state, obs

    def _forcast(self, horizon=1, hist_obs=None, hist_act=None,
                 nxt_act=None, stoch=False, average=False, seed=None):

        assert hist_obs.shape[0] >= self.obs_lag

        npr.seed(seed)

        hist_obs = np.atleast_2d(hist_obs)
        hist_act = np.atleast_2d(hist_act) if hist_act is not None else None

        nxt_act = np.zeros((horizon, self.act_dim)) if nxt_act is None else nxt_act
        nxt_obs = np.zeros((self.obs_lag + horizon, self.obs_dim))
        nxt_state = np.zeros((self.obs_lag + horizon, ), np.int64)

        alpha = self.filtered_state(hist_obs, hist_act)

        if average:
            # empty discrete state when averaging
            nxt_state = None

            for t in range(self.obs_lag):
                nxt_obs[t] = hist_obs[-self.obs_lag:][t]

            # take last belief state from filter
            belief = alpha[-1]
            for t in range(self.obs_lag, self.obs_lag + horizon):
                # average over transitions and belief space
                xl, ul = nxt_obs[t - 1], nxt_act[t - self.obs_lag]
                trans = self.transitions.matrix(xl, ul)

                # update belief with transition
                belief = trans.T @ belief
                belief /= belief.sum()

                # average observations
                for k in range(self.nb_states):
                    xr = nxt_obs[t - self.obs_lag:t]
                    ul = nxt_act[t - self.obs_lag]
                    nxt_obs[t] += belief[k] * (self.observations.sample(k, xr, ul, ar=True) if stoch
                                               else self.observations.mean(k, xr, ul, ar=True))
        else:
            for t in range(self.obs_lag):
                nxt_state[t] = npr.choice(self.nb_states, p=alpha[-self.obs_lag:][t]) if stoch\
                               else np.argmax(alpha[-self.obs_lag:][t])
                nxt_obs[t] = hist_obs[-self.obs_lag:][t]

            for t in range(self.obs_lag, self.obs_lag + horizon):
                zl = nxt_state[t - 1]
                xl = nxt_obs[t - 1]
                ul = nxt_act[t - self.obs_lag]
                nxt_state[t] = self.transitions.sample(zl, xl, ul) if stoch\
                               else self.transitions.likeliest(zl, xl, ul)

                zn = nxt_state[t]
                xr = nxt_obs[t - self.obs_lag:t]
                nxt_obs[t] = self.observations.sample(zn, xr, ul, ar=True) if stoch\
                             else self.observations.mean(zn, xr, ul, ar=True)

        return nxt_state, nxt_obs

    def forcast(self, horizon=None, hist_obs=None, hist_act=None,
                nxt_act=None, stoch=False, average=False, nodes=8):

        if hist_act is None:
            assert nxt_act is None

        hist_act = [None] * len(horizon) if hist_act is None else hist_act
        nxt_act = [None] * len(horizon) if nxt_act is None else nxt_act
        seeds = [i for i in range(len(horizon))]

        nxt_state, nxt_obs = [], []
        for hr, hobs, hact, nact, seed in zip(horizon, hist_obs, hist_act, nxt_act, seeds):
            _nxt_state, _nxt_obs = self._forcast(hr, hobs, hact, nact, stoch, average, seed)
            nxt_state.append(_nxt_state)
            nxt_obs.append(_nxt_obs)

        return nxt_state, nxt_obs

    def _kstep_error(self, obs, act, horizon=1, stoch=False, average=False):

        from sklearn.metrics import mean_squared_error, \
            explained_variance_score, r2_score

        hist_obs, hist_act, nxt_act = [], [], []
        forcast, target, prediction = [], [], []

        nb_steps = obs.shape[0] - horizon - self.obs_lag + 1
        for t in range(nb_steps):
            hist_obs.append(obs[:t + self.obs_lag])
            hist_act.append(act[:t + self.obs_lag])
            nxt_act.append(act[t + self.obs_lag - 1:t + self.obs_lag - 1 + horizon])

        hr = [horizon for _ in range(nb_steps)]
        forcast = self.forcast(horizon=hr, hist_obs=hist_obs, hist_act=hist_act,
                               nxt_act=nxt_act, stoch=stoch, average=average)[1]

        for t in range(nb_steps):
            target.append(obs[t + self.obs_lag - 1 + horizon])
            prediction.append(forcast[t][-1])

        target = np.vstack(target)
        prediction = np.vstack(prediction)

        mse = mean_squared_error(target, prediction)
        smse = 1. - r2_score(target, prediction, multioutput='variance_weighted')
        evar = explained_variance_score(target, prediction, multioutput='variance_weighted')

        return mse, smse, evar

    @ensure_args_are_viable
    def kstep_error(self, obs, act, horizon=1, stoch=False, average=False):

        mse, smse, evar = [], [], []
        for _obs, _act in zip(obs, act):
            _mse, _smse, _evar = self._kstep_error(_obs, _act, horizon, stoch, average)
            mse.append(_mse)
            smse.append(_smse)
            evar.append(_evar)

        return np.mean(mse), np.mean(smse), np.mean(evar)
