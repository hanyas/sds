import autograd.numpy as np
import autograd.numpy.random as npr

from scipy.special import logsumexp

from sds import HMM

from sds.transitions import RecurrentTransition, RecurrentOnlyTransition, NeuralRecurrentTransition, NeuralRecurrentOnlyTransition
from sds.observations import GaussianObservation, AutoRegressiveGaussianObservation, AutoRegressiveGaussianExtendedObservation

from sds.utils import linear_regression


class erARHMM(HMM):

    def __init__(self, nb_states, dm_obs, dm_act, type='recurrent'):
        super(erARHMM, self).__init__(nb_states, dm_obs, dm_act)

        self.type = type

        # init observation
        self.init_observation = GaussianObservation(nb_states=1, dm_obs=self.dm_obs, dm_act=dm_act)

        # transitions
        if self.type == 'recurrent':
            self.transitions = RecurrentTransition(self.nb_states, self.dm_obs, self.dm_act, degree=1)
        elif self.type == 'recurrent-only':
            self.transitions = RecurrentOnlyTransition(self.nb_states, self.dm_obs, self.dm_act, degree=1)
        elif self.type == 'neural-recurrent':
            self.transitions = NeuralRecurrentTransition(self.nb_states, self.dm_obs, self.dm_act, hidden_layer_sizes=(10,))
        elif self.type == 'neural-recurrent-only':
            self.transitions = NeuralRecurrentOnlyTransition(self.nb_states, self.dm_obs, self.dm_act, hidden_layer_sizes=(10, ))

        # observations
        self.observations = AutoRegressiveGaussianExtendedObservation(self.nb_states, self.dm_obs, self.dm_act)

        self.loglikhds = None

    def sample(self, T, act=None):
        act = []
        obs = []
        state = []

        N = len(T)
        for n in range(N):
            _act = np.zeros((T[n], self.dm_act))
            _obs = np.zeros((T[n], self.dm_obs))
            _state = np.zeros((T[n], ), np.int64)

            _state[0] = self.init_state.sample()
            _obs[0, :] = self.init_observation.sample(z=0)
            for t in range(1, T[n]):
                _act[t - 1, :] = self.observations.sample_u(_state[t - 1], _obs[t - 1, :])
                _state[t] = self.transitions.sample(_state[t - 1], _obs[t - 1, :], _act[t - 1, :])
                _obs[t, :] = self.observations.sample_x(_state[t], _obs[t - 1, :], _act[t - 1, :])

            _act[-1, :] = self.observations.sample_u(_state[-1], _obs[-1, :])

            state.append(_state)
            obs.append(_obs)
            act.append(_act)

        return state, obs, act

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

        self.observations.cov_x = aux

        aux = np.zeros((self.nb_states, self.dm_act, self.dm_act))
        for k in range(self.nb_states):
            ts = [np.where(z == k)[0] for z in zs]
            xs = [_obs[t, :] for t, _obs in zip(ts, obs)]
            ys = [_act[t, :] for t, _act in zip(ts, act)]

            coef_, intercept_, sigmas = linear_regression(xs, ys)
            self.observations.K[k, ...] = coef_[:, :self.dm_act]
            self.observations.kff[k, :] = intercept_
            aux[k, ...] = np.diag(sigmas)

        self.observations.cov_u = aux

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
            armu = np.array([self.observations.mean_x(k, _obs[:-1, :], _act[:-1, :self.dm_act]) for k in range(self.nb_states)])
            _mean.append(np.einsum('nk,knl->nl', _gamma, np.concatenate((imu, armu), axis=1)))

        return _mean
