import autograd.numpy as np
import autograd.numpy.random as npr

from scipy.special import logsumexp

from inf.sds.distributions import CategoricalInitState, GaussianObservation, StationaryTransition
from inf.sds.util import normalize, permutation

from inf.sds.cythonized.hmm_cy import filter_cy, smooth_cy

from autograd.tracer import getval
to_c = lambda arr: np.copy(getval(arr), 'C') if not arr.flags['C_CONTIGUOUS'] else getval(arr)


class HMM:

    def __init__(self, nb_states, dim_obs):
        self.nb_states = nb_states
        self.dim_obs = dim_obs

        # init state
        self.init_state = CategoricalInitState(self.nb_states)

        # transitions
        self.transitions = StationaryTransition(self.nb_states)

        # observations
        self.observations = GaussianObservation(self.nb_states, self.dim_obs)

        self.loglikhds = None

    def sample(self, T):
        obs = []
        state = []

        N = len(T)
        for n in range(N):
            _obs = np.zeros((T[n], self.dim_obs))
            _state = np.zeros((T[n], ), np.int64)

            _state[0] = self.init_state.sample()
            for t in range(T[n] - 1):
                _obs[t, :] = self.observations.sample(_state[t])
                _state[t + 1] = self.transitions.sample(_state[t])

            _obs[-1, :] = self.observations.sample(_state[-1])

            state.append(_state)
            obs.append(_obs)

        return state, obs

    def initialize(self, obs):
        from sklearn.cluster import KMeans
        _obs = np.concatenate(obs)
        km = KMeans(self.nb_states).fit(_obs)

        self.observations.mu = km.cluster_centers_
        self.observations.cov = np.array([np.cov(_obs[km.labels_ == k].T)
                                          for k in range(self.nb_states)])

    def log_priors(self):
        logprior = 0.0
        logprior += self.init_state.log_prior()
        logprior += self.transitions.log_prior()
        logprior += self.observations.log_prior()
        return logprior

    def log_likelihoods(self, obs):
        loginit = self.init_state.log_likelihood()
        logtrans = self.transitions.log_likelihood()
        logobs = self.observations.log_likelihood(obs)
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

    def viterbi(self, obs):
        loginit, logtrans, logobs = self.log_likelihoods(obs)

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

    def estep(self, obs):
        self.loglikhds = self.log_likelihoods(obs)
        alpha = self.filter(self.loglikhds)
        beta = self.smooth(self.loglikhds)
        gamma = self.expectations(alpha, beta)
        zeta = self.two_slice(self.loglikhds, alpha, beta)

        return gamma, zeta

    def mstep(self, obs, gamma, zeta):
        self.init_state.mstep([_gamma[0, :] for _gamma in gamma])
        self.transitions.mstep(zeta)
        self.observations.mstep(obs, gamma)

    def em(self, obs, nb_iter=50, prec=1e-6, verbose=False):
        lls = []
        last_ll = - np.inf

        it = 0
        while it < nb_iter:
            gamma, zeta = self.estep(obs)

            ll = self.log_probability(obs)
            lls.append(ll)
            if verbose:
                print("it=", it, "ll=", ll)

            if (ll - last_ll) < prec:
                break
            else:
                self.mstep(obs, gamma, zeta)
                last_ll = ll

            it += 1

        return lls

    def permute(self, perm):
        self.init_state.permute(perm)
        self.transitions.permute(perm)
        self.observations.permute(perm)

    def log_norm(self, obs):
        if self.loglikhds is None:
            self.loglikhds = self.log_likelihoods(obs)
        alpha = self.filter(self.loglikhds)
        return sum([logsumexp(_alpha[-1, :]) for _alpha in alpha])

    def log_probability(self, obs):
        return self.log_norm(obs) + self.log_priors()

    def mean_observation(self, obs):
        loglikhds = self.log_likelihoods(obs)
        alpha = self.filter(loglikhds)
        beta = self.smooth(loglikhds)
        gamma = self.expectations(alpha, beta)

        return [np.einsum('nk,km->nm', _gamma, self.observations.mu) for _gamma in gamma]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from hips.plotting.colormaps import gradient_cmap
    import seaborn as sns

    sns.set_style("white")
    sns.set_context("talk")

    color_names = [
        "windows blue",
        "red",
        "amber",
        "faded green",
        "dusty purple",
        "orange"
    ]

    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors)

    np.set_printoptions(precision=5, suppress=True)

    true_hmm = HMM(nb_states=3, dim_obs=2)

    thetas = np.linspace(0, 2 * np.pi, true_hmm.nb_states, endpoint=False)
    for k in range(true_hmm.nb_states):
        true_hmm.observations.mu[k, :] = 3 * np.array([np.cos(thetas[k]), np.sin(thetas[k])])

    # trajectory lengths
    T = [95, 85, 75]

    true_z, y = true_hmm.sample(T=T)
    true_ll = true_hmm.log_probability(y)

    hmm = HMM(nb_states=3, dim_obs=2)
    hmm.initialize(y)

    lls = hmm.em(y, nb_iter=50, prec=1e-24, verbose=True)
    print("true_ll=", true_ll, "hmm_ll=", lls[-1])

    plt.figure(figsize=(5, 5))
    plt.plot(np.ones(len(lls)) * true_ll, '-r')
    plt.plot(lls)
    plt.show()

    _seq = np.random.choice(len(y))
    hmm.permute(permutation(true_z[_seq], hmm.viterbi([y[_seq]])[1][0]))
    _, hmm_z = hmm.viterbi([y[_seq]])

    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.imshow(true_z[_seq][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    plt.xlim(0, len(y[_seq]))
    plt.ylabel("$z_{\\mathrm{true}}$")
    plt.yticks([])

    plt.subplot(212)
    plt.imshow(hmm_z[0][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    plt.xlim(0, len(y[_seq]))
    plt.ylabel("$z_{\\mathrm{inferred}}$")
    plt.yticks([])
    plt.xlabel("time")

    plt.tight_layout()
    plt.show()

    hmm_y = hmm.mean_observation(y)

    plt.figure(figsize=(8, 4))
    plt.plot(y[_seq] + 10 * np.arange(hmm.dim_obs), '-k', lw=2)
    plt.plot(hmm_y[_seq] + 10 * np.arange(hmm.dim_obs), '-', lw=2)
    plt.show()
