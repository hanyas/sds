import autograd.numpy as np
import autograd.numpy.random as npr

from scipy.special import logsumexp

from sds.transitions import StationaryTransition
from sds.initial import CategoricalInitState
from sds.observations import GaussianObservation, AutoRegressiveGaussianObservation

from sds.utils import linear_regression

from sds.cython.arhmm_cy import filter_cy, smooth_cy

from autograd.tracer import getval
to_c = lambda arr: np.copy(getval(arr), 'C') if not arr.flags['C_CONTIGUOUS'] else getval(arr)


class ARHMM:

    def __init__(self, nb_states, dm_obs, dm_act=0):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        # init state
        self.init_state = CategoricalInitState(self.nb_states)

        # transitions
        self.transitions = StationaryTransition(self.nb_states)

        # init observation
        self.init_observation = GaussianObservation(nb_states=1, dm_obs=self.dm_obs)

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

    def log_priors(self):
        logprior = 0.0
        logprior += self.init_state.log_prior()
        logprior += self.transitions.log_prior()
        logprior += self.observations.log_prior()
        return logprior

    def log_likelihoods(self, obs, act):
        loginit = self.init_state.log_likelihood()
        logtrans = self.transitions.log_likelihood()

        ilog = self.init_observation.log_likelihood([_obs[0, :] for _obs in obs])
        arlog = self.observations.log_likelihood(obs, act)

        logobs = []
        for _ilog, _arlog in zip(ilog, arlog):
            logobs.append(np.vstack((np.repeat(_ilog, self.nb_states), _arlog)))

        return [loginit, logtrans, logobs]

    def filter(self, loglikhds, cython=True):
        loginit, logtrans, logobs = loglikhds

        alpha = []
        for _logobs in logobs:
            T = _logobs.shape[0]
            _alpha = np.zeros((T, self.nb_states))

            if cython:
                filter_cy(to_c(loginit), to_c(logtrans), to_c(_logobs), _alpha)
            else:
                for k in range(self.nb_states):
                    _alpha[0, k] = loginit[k] + _logobs[0, k]

                _aux = np.zeros((self.nb_states,))
                for t in range(1, T):
                    for k in range(self.nb_states):
                        for j in range(self.nb_states):
                            _aux[j] = _alpha[t - 1, j] + logtrans[j, k]
                        _alpha[t, k] = logsumexp(_aux) + _logobs[t, k]

            alpha.append(_alpha)
        return alpha

    def smooth(self, loglikhds, cython=True):
        loginit, logtrans, logobs = loglikhds

        beta = []
        for _logobs in logobs:
            T = _logobs.shape[0]
            _beta = np.zeros((T, self.nb_states))

            if cython:
                smooth_cy(to_c(loginit), to_c(logtrans), to_c(_logobs), _beta)
            else:
                for k in range(self.nb_states):
                    _beta[T - 1, k] = 0.0

                _aux = np.zeros((self.nb_states,))
                for t in range(T - 2, -1, -1):
                    for k in range(self.nb_states):
                        for j in range(self.nb_states):
                            _aux[j] = logtrans[k, j] + _beta[t + 1, j] + _logobs[t + 1, j]
                        _beta[t, k] = logsumexp(_aux)

            beta.append(_beta)
        return beta

    def expectations(self, alpha, beta):
        return [np.exp(_alpha + _beta - logsumexp(_alpha + _beta, axis=1,  keepdims=True)) for _alpha, _beta in zip(alpha, beta)]

    def two_slice(self, loglikhds, alpha, beta):
        loginit, logtrans, logobs = loglikhds

        zeta = []
        for _logobs, _alpha, _beta in zip(logobs, alpha, beta):
            _zeta = _alpha[:-1, :, None] + _beta[1:, None, :] +\
                    _logobs[1:, :][:, None, :] + logtrans

            _zeta -= _zeta.max((1, 2))[:, None, None]
            _zeta = np.exp(_zeta)
            _zeta /= _zeta.sum((1, 2))[:, None, None]

            zeta.append(_zeta)
        return zeta

    def viterbi(self, obs, act):
        loginit, logtrans, logobs = self.log_likelihoods(obs, act)

        delta = []
        z = []
        for _logobs in logobs:
            T = _logobs.shape[0]

            _delta = np.zeros((T, self.nb_states))
            _args = np.zeros((T, self.nb_states), np.int64)
            _z = np.zeros((T, ), np.int64)

            _aux = np.zeros((self.nb_states,))
            for k in range(self.nb_states):
                _aux[k] = _logobs[0, k] + loginit[k]

            _delta[0, :] = np.max(_aux, axis=0)
            _args[0, :] = np.argmax(_delta[0, :], axis=0)

            for t in range(1, T):
                for j in range(self.nb_states):
                    for i in range(self.nb_states):
                        _aux[i] = _delta[t - 1, i] + logtrans[i, j] + _logobs[t, j]

                    _delta[t, j] = np.max(_aux, axis=0)
                    _args[t, j] = np.argmax(_aux, axis=0)

            # backtrace
            _z[T - 1] = np.argmax(_delta[T - 1, :], axis=0)
            for t in range(T - 2, -1, -1):
                _z[t] = _args[t + 1, _z[t + 1]]

            delta.append(_delta)
            z.append(_z)

        return delta, z

    def estep(self, obs, act):
        self.loglikhds = self.log_likelihoods(obs, act)
        alpha = self.filter(self.loglikhds)
        beta = self.smooth(self.loglikhds)
        gamma = self.expectations(alpha, beta)
        zeta = self.two_slice(self.loglikhds, alpha, beta)

        return gamma, zeta

    def mstep(self, obs, act, gamma, zeta):
        self.init_state.mstep([_gamma[0, :] for _gamma in gamma])
        self.transitions.mstep(zeta)
        self.observations.mstep(obs, act, gamma)

    def em(self, obs, act, nb_iter=50, prec=1e-6, verbose=False):
        lls = []
        last_ll = - np.inf

        it = 0
        while it < nb_iter:
            gamma, zeta = self.estep(obs, act)

            ll = self.log_probability(obs, act)
            lls.append(ll)
            if verbose:
                print("it=", it, "ll=", ll)

            if (ll - last_ll) < prec:
                break
            else:
                self.mstep(obs, act, gamma, zeta)
                last_ll = ll

            it += 1

        return lls

    def permute(self, perm):
        self.init_state.permute(perm)
        self.transitions.permute(perm)
        self.observations.permute(perm)

    def log_norm(self, obs, act):
        if self.loglikhds is None:
            self.loglikhds = self.log_likelihoods(obs, act)
        alpha = self.filter(self.loglikhds)
        return sum([logsumexp(_alpha[-1, :]) for _alpha in alpha])

    def log_probability(self, obs, act):
        return self.log_norm(obs, act) + self.log_priors()

    def mean_observation(self, obs, act):
        loglikhds = self.log_likelihoods(obs, act)
        alpha = self.filter(loglikhds)
        beta = self.smooth(loglikhds)
        gamma = self.expectations(alpha, beta)

        imu = np.array([self.init_observation.mu for _ in range(self.nb_states)])

        _mean = []
        for _obs, _act, _gamma in zip(obs, act, gamma):
            armu = np.array([self.observations.mean(k, _obs[:-1, :], _act[:-1, :self.dm_act]) for k in range(self.nb_states)])
            _mean.append(np.einsum('nk,knl->nl', _gamma, np.concatenate((imu, armu), axis=1)))

        return _mean
