import autograd.numpy as np
import autograd.numpy.random as npr

from sds import HMM
from sds.observations import GaussianObservation, AutoRegressiveGaussianObservation

from sds.utils import linear_regression


class ARHMM(HMM):

    def __init__(self, nb_states, dm_obs, dm_act=0):
        super(ARHMM, self).__init__(nb_states, dm_obs, dm_act)

        # init observation
        self.init_observation = GaussianObservation(nb_states=1, dm_obs=self.dm_obs, dm_act=self.dm_act)

        # observations
        self.observations = AutoRegressiveGaussianObservation(self.nb_states, self.dm_obs, self.dm_act)

        self.loglikhds = None

    def sample(self, T, act):
        obs = []
        state = []

        N = len(T)
        for n in range(N):
            _act = act[n]
            _obs = np.zeros((T[n], self.dm_obs))
            _state = np.zeros((T[n], ), np.int64)

            _state[0] = self.init_state.sample()
            _obs[0, :] = self.init_observation.sample(z=0)
            for t in range(1, T[n]):
                _state[t] = self.transitions.sample(_state[t - 1])
                _obs[t, :] = self.observations.sample(_state[t], _obs[t - 1, :], _act[t - 1, :])

            state.append(_state)
            obs.append(_obs)

        return state, obs

    def initialize(self, obs, act, localize=True):
        self.init_observation.mu = npr.randn(1, self.dm_obs)
        self.init_observation.cov = np.array([np.eye(self.dm_obs, self.dm_obs)])

        Ts = [_obs.shape[0] for _obs in obs]
        if localize:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.hstack((np.vstack(obs), np.vstack(act))))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
            zs = [z[:-1] for z in zs]
        else:
            zs = [npr.choice(self.nb_states, size=T - 1) for T in Ts]

        aux = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        for k in range(self.nb_states):
            ts = [np.where(z == k)[0] for z in zs]
            xs = [np.hstack((_obs[t, :], _act[t, :])) for t, _obs, _act in zip(ts, obs, act)]
            ys = [_obs[t + 1, :] for t, _obs in zip(ts, obs)]

            coef_, intercept_, sigmas = linear_regression(xs, ys)
            self.observations.A[k, ...] = coef_[:, :self.dm_obs]
            self.observations.B[k, ...] = coef_[:, self.dm_obs:]
            self.observations.c[k, :] = intercept_
            aux[k, ...] = np.diag(sigmas)

        self.observations.cov = aux

    def log_likelihoods(self, obs, act):
        loginit = self.init_state.log_likelihood()
        logtrans = self.transitions.log_likelihood(obs, act)

        ilog = self.init_observation.log_likelihood([_obs[0, :] for _obs in obs])
        arlog = self.observations.log_likelihood(obs, act)

        logobs = []
        for _ilog, _arlog in zip(ilog, arlog):
            logobs.append(np.vstack((np.repeat(_ilog, self.nb_states), _arlog)))

        return [loginit, logtrans, logobs]

    def mean_observation(self, obs, act):
        loglikhds = self.log_likelihoods(obs, act)
        alpha = self.filter(loglikhds)
        beta = self.smooth(loglikhds)
        gamma = self.marginals(alpha, beta)

        imu = np.array([self.init_observation.mu for _ in range(self.nb_states)])

        _mean = []
        for _obs, _act, _gamma in zip(obs, act, gamma):
            armu = np.array([self.observations.mean(k, _obs[:-1, :], _act[:-1, :self.dm_act]) for k in range(self.nb_states)])
            _mean.append(np.einsum('nk,knl->nl', _gamma, np.concatenate((imu, armu), axis=1)))

        return _mean
