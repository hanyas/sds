import numpy as np
import numpy.random as npr

from scipy.special import logsumexp

from sds.models import HMM

from sds.initial import InitGaussianObservation
from sds.initial import BayesianInitGaussianObservation
from sds.initial import BayesianInitDiagonalGaussianObservation

from sds.observations import AutoRegressiveGaussianObservation
from sds.observations import BayesianAutoRegressiveGaussianObservation
from sds.observations import BayesianAutoRegressiveDiagonalGaussianObservation
from sds.observations import BayesianAutoRegressiveTiedGaussianObservation
from sds.observations import BayesianAutoRegressiveTiedDiagonalGaussianObservation

from sds.utils.decorate import ensure_args_are_viable_lists

from pathos.multiprocessing import ProcessPool


class ARHMM(HMM):

    def __init__(self, nb_states, obs_dim, act_dim=0, nb_lags=1,
                 algo_type='MAP', init_obs_type='full', obs_type='full',
                 init_state_prior={}, init_obs_prior={}, trans_prior={}, obs_prior={},
                 init_state_kwargs={}, init_obs_kwargs={}, trans_kwargs={}, obs_kwargs={}):

        super(ARHMM, self).__init__(nb_states, obs_dim, act_dim,
                                    init_state_kwargs=init_state_kwargs,
                                    trans_kwargs=trans_kwargs)

        self.nb_lags = nb_lags

        if algo_type == 'ML':
            self.init_observation = InitGaussianObservation(self.nb_states, self.obs_dim,
                                                            self.act_dim, self.nb_lags, **init_obs_kwargs)
            self.observations = AutoRegressiveGaussianObservation(self.nb_states, self.obs_dim,
                                                                  self.act_dim, self.nb_lags, **obs_kwargs)
        else:
            if init_obs_type == 'full':
                self.init_observation = BayesianInitGaussianObservation(self.nb_states, self.obs_dim, self.act_dim,
                                                                        self.nb_lags, prior=init_obs_prior,
                                                                        **init_obs_kwargs)
            elif init_obs_type == 'diagonal':
                self.init_observation = BayesianInitDiagonalGaussianObservation(self.nb_states, self.obs_dim, self.act_dim,
                                                                                self.nb_lags, prior=init_obs_prior,
                                                                                **init_obs_kwargs)
            else:
                raise NotImplementedError

            if obs_type == 'full':
                self.observations = BayesianAutoRegressiveGaussianObservation(self.nb_states, self.obs_dim, self.act_dim,
                                                                              self.nb_lags, prior=obs_prior, **obs_kwargs)
            elif obs_type == 'diagonal':
                self.observations = BayesianAutoRegressiveDiagonalGaussianObservation(self.nb_states, self.obs_dim, self.act_dim,
                                                                                      self.nb_lags, prior=obs_prior, **obs_kwargs)
            elif obs_type == 'tied-full':
                self.observations = BayesianAutoRegressiveTiedGaussianObservation(self.nb_states, self.obs_dim, self.act_dim,
                                                                                  self.nb_lags, prior=obs_prior, **obs_kwargs)
            elif obs_type == 'tied-diagonal':
                self.observations = BayesianAutoRegressiveTiedDiagonalGaussianObservation(self.nb_states, self.obs_dim, self.act_dim,
                                                                                          self.nb_lags, prior=obs_prior, **obs_kwargs)
            else:
                raise NotImplementedError

    @ensure_args_are_viable_lists
    def log_likelihoods(self, obs, act=None):
        loginit = self.init_state.log_init()
        logtrans = self.transitions.log_transition(obs, act)

        ilog = self.init_observation.log_likelihood(obs)
        arlog = self.observations.log_likelihood(obs, act)

        logobs = []
        for _ilog, _arlog in zip(ilog, arlog):
            logobs.append(np.vstack((_ilog, _arlog)))

        return loginit, logtrans, logobs

    @ensure_args_are_viable_lists
    def mean_observation(self, obs, act=None):
        loglikhds = self.log_likelihoods(obs, act)
        alpha, norm = self.forward(*loglikhds)
        beta = self.backward(*loglikhds, scale=norm)
        gamma = self.posterior(alpha, beta)

        imu = self.init_observation.smooth(gamma, obs)
        armu = self.observations.smooth(gamma, obs, act)

        mean_obs = []
        for _imu, _armu in zip(imu, armu):
            mean_obs.append(np.vstack((_imu, _armu)))

        return mean_obs

    def _sample(self, horizon, act=None, seed=None):
        npr.seed(seed)

        act = np.zeros((horizon, self.act_dim)) if act is None else act
        obs = np.zeros((horizon, self.obs_dim))
        state = np.zeros((horizon,), np.int64)

        state[0] = self.init_state.sample()
        obs[0, :] = self.init_observation.sample(state[0])

        for t in range(1, self.nb_lags):
            state[t] = self.transitions.sample(state[t - 1], obs[t - 1, :], act[t - 1, :])
            obs[t, :] = self.init_observation.sample(state[t])

        for t in range(self.nb_lags, horizon):
            state[t] = self.transitions.sample(state[t - 1], obs[t - 1, :], act[t - 1, :])
            obs[t, :] = self.observations.sample(state[t], obs[t - self.nb_lags:t, :], act[t - 1, :])

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

        assert hist_obs.shape[0] >= self.nb_lags

        npr.seed(seed)

        hist_obs = np.atleast_2d(hist_obs)
        hist_act = np.atleast_2d(hist_act) if hist_act is not None else None

        nxt_act = np.zeros((horizon, self.act_dim)) if nxt_act is None else nxt_act
        nxt_obs = np.zeros((self.nb_lags + horizon, self.obs_dim))
        nxt_state = np.zeros((self.nb_lags + horizon, ), np.int64)

        belief = self.filter(hist_obs, hist_act)[0]

        if stoch:
            for t in range(self.nb_lags):
                nxt_state[t] = npr.choice(self.nb_states, p=belief[-self.nb_lags:][t])
                nxt_obs[t, :] = hist_obs[-self.nb_lags:][t]

            for t in range(self.nb_lags, self.nb_lags + horizon):
                nxt_state[t] = self.transitions.sample(nxt_state[t - 1],
                                                       nxt_obs[t - 1, :],
                                                       nxt_act[t - self.nb_lags, :])
                nxt_obs[t, :] = self.observations.sample(nxt_state[t],
                                                         nxt_obs[t - self.nb_lags:t, :],
                                                         nxt_act[t - self.nb_lags, :])
        else:
            if average:
                # return empty discrete state when mixing
                nxt_state = None
                for t in range(self.nb_lags):
                    nxt_obs[t, :] = hist_obs[-self.nb_lags:][t]

                # take last belief state from filter
                alpha = belief[-1]
                for t in range(self.nb_lags, self.nb_lags + horizon):
                    # average over transitions and belief space
                    logtrans = np.squeeze(self.transitions.log_transition(nxt_obs[t - 1, :],
                                                                          nxt_act[t - self.nb_lags, :])[0])
                    trans = np.exp(logtrans - logsumexp(logtrans, axis=1, keepdims=True))

                    # update belief
                    alpha = trans.T @ alpha
                    alpha /= alpha.sum()

                    # average observations
                    for k in range(self.nb_states):
                        nxt_obs[t, :] += alpha[k] * self.observations.mean(k,
                                                                           nxt_obs[t - self.nb_lags:t, :],
                                                                           nxt_act[t - self.nb_lags, :])
            else:
                for t in range(self.nb_lags):
                    nxt_state[t] = np.argmax(belief[-self.nb_lags:][t])
                    nxt_obs[t, :] = hist_obs[-self.nb_lags:][t]

                for t in range(self.nb_lags, self.nb_lags + horizon):
                    nxt_state[t] = self.transitions.likeliest(nxt_state[t - 1],
                                                              nxt_obs[t - 1, :],
                                                              nxt_act[t - self.nb_lags, :])
                    nxt_obs[t, :] = self.observations.mean(nxt_state[t],
                                                           nxt_obs[t - self.nb_lags:t, :],
                                                           nxt_act[t - self.nb_lags, :])

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

        nb_steps = obs.shape[0] - horizon - self.nb_lags + 1
        for t in range(nb_steps):
            hist_obs.append(obs[:t + self.nb_lags, :])
            hist_act.append(act[:t + self.nb_lags, :])
            nxt_act.append(act[t + self.nb_lags - 1:t + self.nb_lags - 1 + horizon, :])

        hr = [horizon for _ in range(nb_steps)]
        _, forcast = self.forcast(horizon=hr, hist_obs=hist_obs, hist_act=hist_act,
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
